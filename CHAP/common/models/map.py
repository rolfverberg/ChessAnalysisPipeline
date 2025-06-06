"""Map related Pydantic model classes."""

# System modules
from copy import deepcopy
from functools import (
    cache,
    lru_cache,
)
import os
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party modules
import numpy as np
from pydantic import (
    Field,
    FilePath,
    PrivateAttr,
    conint,
    conlist,
    constr,
    field_validator,
    model_validator,
)
from pyspec.file.spec import FileSpec
from typing_extensions import Annotated

# Local modules
from CHAP.models import CHAPBaseModel


class Detector(CHAPBaseModel):
    """Class representing a single detector.

    :ivar id: The detector id (e.g. name or channel index).
    :type id: str
    :ivar attrs: Additional detector configuration attributes.
    :type attrs: dict, optional
    """
    id: constr(min_length=1)
    attrs: Optional[Annotated[dict, Field(validate_default=True)]] = {}

    @field_validator('id', mode='before')
    @classmethod
    def validate_id(cls, id):
        """Validate the detector id.

        :param id: The detector id (e.g. name or channel index).
        :type id: int, str
        :return: The detector id.
        :rtype: str
        """
        if isinstance(id, int):
            return str(id)
        return id

    #RV maybe better to use model_validator, see v2 docs?
    @field_validator('attrs')
    @classmethod
    def validate_attrs(cls, attrs):
        """Validate any additional detector configuration attributes.

        :param attrs: Any additional attributes to `DetectorConfig`.
        :type attrs: dict
        :raises ValueError: Invalid attribute.
        :return: The validated field for `attrs`.
        :rtype: dict
        """
        # RV FIX add eta
        name = attrs.get('name')
        if name is not None:
            if isinstance(name, int):
                attrs['name'] = str(name)
            elif not isinstance(name, str):
                raise ValueError
        return attrs


class DetectorConfig(CHAPBaseModel):
    """Class representing a detector configuration.

    :ivar detectors: The detector list.
    :type detectors: list[Detector]
    """
    detectors: conlist(item_type=Detector, min_length=1)


class Sample(CHAPBaseModel):
    """Class representing a sample metadata configuration.

    :ivar name: The name of the sample.
    :type name: str
    :ivar description: A description of the sample.
    :type description: str, optional
    """
    name: constr(min_length=1)
    description: Optional[str] = ''


class SpecScans(CHAPBaseModel):
    """Class representing a set of scans from a single SPEC file.

    :ivar spec_file: Path to the SPEC file.
    :type spec_file: str
    :ivar scan_numbers: List of scan numbers to use.
    :type scan_numbers: Union(int, list[int], str)
    :ivar par_file: Path to a non-default SMB par file.
    :type par_file: str, optional
    """
    spec_file: FilePath
    scan_numbers: conlist(item_type=conint(gt=0), min_length=1)
    par_file: Optional[FilePath] = None

    @field_validator('spec_file')
    @classmethod
    def validate_spec_file(cls, spec_file):
        """Validate the specified SPEC file.

        :param spec_file: Path to the SPEC file.
        :type spec_file: str
        :raises ValueError: If the SPEC file is invalid.
        :return: Absolute path to the SPEC file.
        :rtype: str
        """
        try:
            spec_file = os.path.abspath(spec_file)
            FileSpec(spec_file)
        except Exception as exc:
            raise ValueError(f'Invalid SPEC file {spec_file}') from exc
        return spec_file

    @field_validator('scan_numbers', mode='before')
    @classmethod
    def validate_scan_numbers(cls, scan_numbers, info):
        """Validate the specified list of scan numbers.

        :param scan_numbers: List of scan numbers.
        :type scan_numbers: Union(int, list[int], str)
        :param info: Pydantic validator info object.
        :type info: pydantic_core._pydantic_core.ValidationInfo
        :raises ValueError: If a specified scan number is not found in
            the SPEC file.
        :return: List of scan numbers.
        :rtype: list[int]
        """
        if isinstance(scan_numbers, int):
            scan_numbers = [scan_numbers]
        elif isinstance(scan_numbers, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_numbers = string_to_list(scan_numbers)

        spec_file = info.data.get('spec_file')
        if spec_file is not None:
            spec_scans = FileSpec(spec_file)
            for scan_number in scan_numbers:
                scan = spec_scans.get_scan_by_number(scan_number)
                if scan is None:
                    raise ValueError(
                        f'No scan number {scan_number} in {spec_file}')
        return scan_numbers

    @field_validator('par_file')
    @classmethod
    def validate_par_file(cls, par_file):
        """Validate the specified SMB par file.

        :param par_file: Path to a non-default SMB par file.
        :type par_file: str
        :raises ValueError: If the SMB par file is invalid.
        :return: Absolute path to the SMB par file.
        :rtype: str
        """
        if par_file is None or not par_file:
            return ''
        par_file = os.path.abspath(par_file)
        if not os.path.isfile(par_file):
            raise ValueError(f'Invalid SMB par file {par_file}')
        return par_file

    @property
    def scanparsers(self):
        """Returns the list of `ScanParser`s for each of the scans
        specified by the SPEC file and scan numbers belonging to this
        instance of `SpecScans`.
        """
        return [self.get_scanparser(scan_no) for scan_no in self.scan_numbers]

    def get_scanparser(self, scan_number):
        """Return a `ScanParser` for the specified scan number in the
        specified SPEC file.

        :param scan_number: Scan number to get a `ScanParser` for.
        :type scan_number: int
        :return: `ScanParser` for the specified scan number.
        :rtype: ScanParser
        """
        if self.par_file:
            return get_scanparser(
                self.spec_file, scan_number, par_file=self.par_file)
        return get_scanparser(self.spec_file, scan_number)

    def get_index(self, scan_number, scan_step_index, map_config):
        """Return a tuple representing the index of a specific step in
        a specific SPEC scan within a map.

        :param scan_number: Scan number to get index for.
        :type scan_number: int
        :param scan_step_index: Scan step index to get index for.
        :type scan_step_index: int
        :param map_config: Map configuration to get index for.
        :type map_config: MapConfig
        :return: Index for the specified scan number and scan step
            index within the specified map configuration.
        :rtype: tuple
        """
        index = ()
        for independent_dimension in map_config.independent_dimensions:
            coordinate_index = list(
                map_config.coords[independent_dimension.label]).index(
                    independent_dimension.get_value(
                        self, scan_number, scan_step_index,
                        map_config.scalar_data))
            index = (coordinate_index, *index)
        return index

    def get_detector_data(self, detectors, scan_number, scan_step_index):
        """Return the raw data from the specified detectors at the
        specified scan number and scan step index.

        :param detectors: List of detector prefixes to get raw data
            for.
        :type detectors: list[str]
        :param scan_number: Scan number to get data for.
        :type scan_number: int
        :param scan_step_index: Scan step index to get data for.
        :type scan_step_index: int
        :return: Data from the specified detectors for the specified
            scan number and scan step index.
        :rtype: list[np.ndarray]
        """
        return get_detector_data(
            tuple([detector.prefix for detector in detectors]),
            self.spec_file,
            scan_number,
            scan_step_index)


@cache
def get_available_scan_numbers(spec_file):
    """Get the available scan numbers.

    :param spec_file: Path to the SPEC file.
    :type spec_file: str
    """
    return list(FileSpec(spec_file).scans.keys())


@cache
def get_scanparser(spec_file, scan_number, par_file=None):
    """Get the scanparser.

    :param spec_file: Path to the SPEC file.
    :type spec_file: str
    :param scan_number: Scan number to get data for.
    :type scan_number: int
    :param par_file: Path to a SMB par file.
    :type par_file: str, optional
    """
    if scan_number not in get_available_scan_numbers(spec_file):
        return None
    if par_file is None:
        return ScanParser(spec_file, scan_number)
    return ScanParser(spec_file, scan_number, par_file=par_file)


@lru_cache(maxsize=10)
def get_detector_data(
        detector_prefixes, spec_file, scan_number, scan_step_index):
    """Get the detector data.

    :param detector_prefixes: The detector prefixes.
    :type detector_prefixes: Union[tuple[str], list[str]]
    :ivar spec_file: Path to the SPEC file.
    :type spec_file: str
    :param scan_number: Scan number to get data for.
    :type scan_number: int
    :param scan_step_index: The scan step index.
    :type scan_step_index: int
    """
    detector_data = []
    scanparser = get_scanparser(spec_file, scan_number)
    for prefix in detector_prefixes:
        image_data = scanparser.get_detector_data(prefix, scan_step_index)
        detector_data.append(image_data)
    return detector_data


class PointByPointScanData(CHAPBaseModel):
    """Class representing a source of raw scalar-valued data for which
    a value was recorded at every point in a `MapConfig`.

    :ivar label: A user-defined label for referring to this data in
        the NeXus file and in other tools.
    :type label: str
    :ivar units: The units in which the data were recorded.
    :type units: str
    :ivar data_type: Represents how these data were recorded at time
        of data collection.
    :type data_type: Literal['spec_motor', 'spec_motor_absolute',
        'scan_column', 'smb_par', 'expression', 'detector_log_timestamps']
    :ivar name: Represents the name with which these raw data were
        recorded at time of data collection.
    :type name: str
    """
    label: constr(min_length=1)
    units: constr(strip_whitespace=True, min_length=1)
    data_type: Literal[
        'spec_motor', 'spec_motor_absolute', 'scan_column', 'smb_par',
        'expression', 'detector_log_timestamps']
    name: constr(strip_whitespace=True, min_length=1)
    ndigits: Optional[conint(ge=0)] = None

    @field_validator('label')
    @classmethod
    def validate_label(cls, label):
        """Validate that the supplied `label` does not conflict with
        any of the values for `label` reserved for certain data needed
        to perform corrections.

        :param label: The value of `label` to validate.
        :type label: str
        :raises ValueError: If `label` is one of the reserved values.
        :return: The originally supplied value `label`.
        :rtype: str
        """
        if ((not issubclass(cls,CorrectionsData))
                and label in CorrectionsData.reserved_labels()):
            raise ValueError(
                f'{cls.__name__}.label may not be any of the following '
                f'reserved values: {CorrectionsData.reserved_labels()}')
        return label

    def validate_for_station(self, station):
        """Validate this instance of `PointByPointScanData` for a
        certain choice of station (beamline).

        :param station: The name of the station (in 'idxx' format).
        :type station: str
        :raises TypeError: If the station is not compatible with the
            value of the `data_type` attribute for this instance of
            PointByPointScanData.
        """
        if (station.lower() not in ('id1a3', 'id3a')
                and self.data_type == 'smb_par'):
            raise TypeError(
                f'{self.__class__.__name__}.data_type may not be "smb_par" '
                f'when station is "{station}"')
        if (not station.lower() == 'id3b'
                and self.data_type == 'detector_log_timestamps'):
            raise TypeError(
                f'{self.__class__.__name__}.data_type may not be'
                + f' "detector_log_timestamps" when station is "{station}"')

    def validate_for_spec_scans(
            self, spec_scans, scan_step_index='all'):
        """Validate this instance of `PointByPointScanData` for a list
        of `SpecScans`.

        :param spec_scans: A list of `SpecScans` whose raw data will
            be checked for the presence of the data represented by
            this instance of `PointByPointScanData`.
        :type spec_scans: list[SpecScans]
        :param scan_step_index: A specific scan step index to validate,
            defaults to `'all'`.
        :type scan_step_index: Union[Literal['all'],int], optional
        :raises RuntimeError: If the data represented by this instance
            of `PointByPointScanData` is missing for the specified
            scan steps.
        """
        for scans in spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                if scan_step_index == 'all':
                    scan_step_index_range = range(scanparser.spec_scan_npts)
                else:
                    scan_step_index_range = range(
                        scan_step_index, 1+scan_step_index)
                for index in scan_step_index_range:
                    try:
                        self.get_value(scans, scan_number, index)
                    except Exception as exc:
                        raise RuntimeError(
                            f'Could not find data for {self.name} '
                            f'(data_type "{self.data_type}") '
                            f'on scan number {scan_number} '
                            f'for index {index} '
                            f'in spec file {scans.spec_file}') from exc

    def validate_for_scalar_data(self, scalar_data):
        """Used for `PointByPointScanData` objects with a `data_type`
        of `'expression'`. Validate that the `scalar_data` field of a
        `MapConfig` object contains all the items necessary for
        evaluating the expression.

        :param scalar_data: the `scalar_data` field of a `MapConfig`
            that this `PointByPointScanData` object will be validated
            against.
        :type scalar_data: list[PointByPointScanData]
        :raises ValueError: if `scalar_data` does not contain items
           needed for evaluating the expression.
        """
        # Third party modules
        from ast import parse
        from asteval import get_ast_names

        labels = get_ast_names(parse(self.name))
        for label in ('round', 'np', 'numpy'):
            try:
                labels.remove(label)
            except:
                pass
        for l in labels:
            if l in ('round', 'np', 'numpy'):
                continue
            label_found = False
            for s_d in scalar_data:
                if s_d.label == l:
                    label_found = True
                    break
            if not label_found:
                raise ValueError(
                    f'{l} is not the label of an item in scalar_data')

    def get_value(
            self, spec_scans, scan_number, scan_step_index=0,
            scalar_data=None, relative=True, ndigits=None):
        """Return the value recorded for this instance of
        `PointByPointScanData` at a specific scan step.

        :param spec_scans: An instance of `SpecScans` in which the
            requested scan step occurs.
        :type spec_scans: SpecScans
        :param scan_number: The number of the scan in which the
            requested scan step occurs.
        :type scan_number: int
        :param scan_step_index: The index of the requested scan step,
            defaults to `0`.
        :type scan_step_index: int, optional
        :param scalar_data: list of scalar data configurations used to
            get values for `PointByPointScanData` objects with
            `data_type == 'expression'`.
        :type scalar_data: list[PointByPointScanData], optional
        :param relative: Whether to return a relative value or not,
            defaults to `True` (only applies to SPEC motor values).
        :type relative: bool, optional
        :params ndigits: Round SPEC motor values to the specified
            number of decimals if set.
        :type ndigits: int, optional
        :return: The value recorded of the data represented by this
            instance of `PointByPointScanData` at the scan step
            requested.
        :rtype: float
        """
        if 'spec_motor' in self.data_type:
            if ndigits is None:
                ndigits = self.ndigits
            if 'absolute' in self.data_type:
                relative = False
            return get_spec_motor_value(
                spec_scans.spec_file, scan_number, scan_step_index, self.name,
                relative, ndigits)
        if self.data_type == 'scan_column':
            return get_spec_counter_value(
                spec_scans.spec_file, scan_number, scan_step_index, self.name)
        if self.data_type == 'smb_par':
            return get_smb_par_value(
                spec_scans.spec_file, scan_number, self.name)
        if self.data_type == 'expression':
            if scalar_data is None:
                scalar_data = []
            return get_expression_value(
                spec_scans, scan_number, scan_step_index, self.name,
                scalar_data)
        if self.data_type == 'detector_log_timestamps':
            timestamps = get_detector_log_timestamps(
                spec_scans.spec_file, scan_number, self.name)
            if scan_step_index >= 0:
                return timestamps[scan_step_index]
            else:
                return timestamps
        return None

@cache
def get_spec_motor_value(
        spec_file, scan_number, scan_step_index, spec_mnemonic,
        relative=True, ndigits=None):
    """Return the value recorded for a SPEC motor at a specific scan
    step.

    :param spec_file: Location of a SPEC file in which the requested
        scan step occurs.
    :type spec_scans: str
    :param scan_number: The number of the scan in which the requested
        scan step occurs.
    :type scan_number: int
    :param scan_step_index: The index of the requested scan step.
    :type scan_step_index: int
    :param spec_mnemonic: The menmonic of a SPEC motor.
    :type spec_mnemonic: str
    :param relative: Whether to return a relative value or not,
        defaults to `True`.
    :type relative: bool, optional
    :params ndigits: Round SPEC motor values to the specified
        number of decimals if set.
    :type ndigits: int, optional
    :return: The value of the motor at the scan step requested.
    :rtype: float
    """
    scanparser = get_scanparser(spec_file, scan_number)
    if (hasattr(scanparser, 'spec_scan_motor_mnes')
            and spec_mnemonic in scanparser.spec_scan_motor_mnes):
        motor_i = scanparser.spec_scan_motor_mnes.index(spec_mnemonic)
        if scan_step_index >= 0:
            scan_step = np.unravel_index(
                scan_step_index,
                scanparser.spec_scan_shape,
                order='F')
            motor_value = \
                scanparser.get_spec_scan_motor_vals(
                    relative)[motor_i][scan_step[motor_i]]
        else:
            motor_value = scanparser.get_spec_scan_motor_vals(
                relative)[motor_i]
            if len(scanparser.spec_scan_shape) == 2:
                if motor_i == 0:
                    motor_value = np.concatenate(
                        [motor_value] * scanparser.spec_scan_shape[1])
                else:
                    motor_value = np.repeat(
                        motor_value, scanparser.spec_scan_shape[0])
    else:
        motor_value = scanparser.get_spec_positioner_value(spec_mnemonic)
    if ndigits is not None:
        motor_value = np.round(motor_value, ndigits)
    return motor_value

@cache
def get_spec_counter_value(
        spec_file, scan_number, scan_step_index, spec_column_label):
    """Return the value recorded for a SPEC counter at a specific scan
    step.

    :param spec_file: Location of a SPEC file in which the requested
        scan step occurs.
    :type spec_scans: str
    :param scan_number: The number of the scan in which the requested
        scan step occurs.
    :type scan_number: int
    :param scan_step_index: The index of the requested scan step.
    :type scan_step_index: int
    :param spec_column_label: The label of a SPEC data column.
    :type spec_column_label: str
    :return: The value of the counter at the scan step requested.
    :rtype: float
    """
    scanparser = get_scanparser(spec_file, scan_number)
    if scan_step_index >= 0:
        return scanparser.spec_scan_data[spec_column_label][scan_step_index]
    return scanparser.spec_scan_data[spec_column_label]

@cache
def get_smb_par_value(spec_file, scan_number, par_name):
    """Return the value recorded for a specific scan in SMB-tyle .par
    file.

    :param spec_file: Location of a SPEC file in which the requested
        scan step occurs.
    :type spec_scans: str
    :param scan_number: The number of the scan in which the requested
        scan step occurs.
    :type scan_number: int
    :param par_name: The name of the column in the .par file.
    :type par_name: str
    :return: The value of the .par file value for  the scan requested.
    :rtype: float
    """
    scanparser = get_scanparser(spec_file, scan_number)
    return scanparser.pars[par_name]

def get_expression_value(
        spec_scans, scan_number, scan_step_index, expression,
        scalar_data):
    """Return the value of an evaluated expression of other sources of
    point-by-point scalar scan data for a single point.

    :param spec_scans: An instance of `SpecScans` in which the
        requested scan step occurs.
    :type spec_scans: SpecScans
    :param scan_number: The number of the scan in which the requested
        scan step occurs.
    :type scan_number: int
    :param scan_step_index: The index of the requested scan step.
    :type scan_step_index: int
    :param expression: The string expression to evaluate.
    :type expression: str
    :param scalar_data: the `scalar_data` field of a `MapConfig`
        object (used to provide values for variables used in
        `expression`).
    :type scalar_data: list[PointByPointScanData]
    :return: The value of the .par file value for  the scan requested.
    :rtype: float
    """
    # Third party modules
    from ast import parse
    from asteval import get_ast_names, Interpreter

    labels = get_ast_names(parse(expression))
    symtable = {}
    for l in labels:
        if l == 'round':
            symtable[l] = round
        for s_d in scalar_data:
            if s_d.label == l:
                symtable[l] = s_d.get_value(
                    spec_scans, scan_number, scan_step_index, scalar_data)
    aeval = Interpreter(symtable=symtable)
    return aeval(expression)

@cache
def get_detector_log_timestamps(spec_file, scan_number, detector_prefix):
    """Return the list of detector timestamps for the given scan &
    detector prefix.

    :param spec_file: Location of a SPEC file in which the requested
        scan occurs.
    :type spec_scans: str
    :param scan_number: The number of the scan for which to return
        detector log timestamps.
    :type scan_number: int
    :param detector_prefix: The prefix of the detecotr whose log file
        should be used.
    :return: All detector log timestamps for the given scan.
    :rtype: list[float]
    """
    sp = get_scanparser(spec_file, scan_number)
    return sp.get_detector_log_timestamps(detector_prefix)

def validate_data_source_for_map_config(data_source, info):
    """Confirm that an instance of PointByPointScanData is valid for
    the station and scans provided by a map configuration dictionary.

    :param data_source: The input object to validate.
    :type data_source: PointByPointScanData
    :param info: Pydantic validator info object.
    :type info: pydantic_core._pydantic_core.ValidationInfo
    :raises Exception: If `data_source` cannot be validated.
    :return: The validated `data_source` instance.
    :rtype: PointByPointScanData
    """
    def _validate_data_source_for_map_config(
            data_source, info):
        if isinstance(data_source, list):
            return [_validate_data_source_for_map_config(d_s, info)
                    for d_s in data_source]
        if data_source is not None:
            values = info.data
            if data_source.data_type == 'expression':
                data_source.validate_for_scalar_data(values['scalar_data'])
            else:
                import_scanparser(
                    values['station'], values['experiment_type'])
                data_source.validate_for_station(values['station'])
                data_source.validate_for_spec_scans(values['spec_scans'])
        return data_source

    return _validate_data_source_for_map_config(data_source, info)


class IndependentDimension(PointByPointScanData):
    """Class representing the source of data to identify the
    coordinate values along one dimension of a `MapConfig`.

    :ivar start: Sarting index for slicing all datasets of a
        `MapConfig` along this axis, defaults to `0`.
    :type start: int, optional
    :ivar end: Ending index for slicing all datasets of a `MapConfig`
        along this axis, defaults to the total number of unique values
        along this axis in the associated `MapConfig`.
    :type end: int, optional
    :ivar step: Step for slicing all datasets of a `MapConfig` along
        this axis, defaults to `1`.
    :type step: int, optional
    """
    start: Optional[conint(ge=0)] = 0
    end: Optional[int] = None
    step: Optional[conint(gt=0)] = 1

#    @field_validator('step')
#    @classmethod
#    def validate_step(cls, step):
#        """Validate that the supplied value of `step`.
#
#        :param step: The value of `step` to validate.
#        :type step: str
#        :raises ValueError: If `step` is zero.
#        :return: The originally supplied value `step`.
#        :rtype: int
#        """
#        if step == 0 :
#            raise ValueError('slice step cannot be zero')
#        return step


class CorrectionsData(PointByPointScanData):
    """Class representing the special instances of
    `PointByPointScanData` that are used by certain kinds of
    `CorrectionConfig` tools.

    :ivar label: One of the reserved values required by
        `CorrectionConfig`.
    :type label: Literal['presample_intensity', 'postsample_intensity',
                         'dwell_time_actual']
    :ivar data_type: Represents how these data were recorded at time
        of data collection.
    :type data_type: Literal['scan_column', 'smb_par']
    """
    label: Literal['presample_intensity', 'postsample_intensity',
                   'dwell_time_actual']
    data_type: Literal['scan_column','smb_par']

    @classmethod
    def reserved_labels(cls):
        """Return a list of all the labels reserved for
        corrections-related scalar data.

        :return: A list of reserved labels.
        :rtype: list[str]
        """
        return list((*cls.model_fields['label'].annotation.__args__, 'round'))


class PresampleIntensity(CorrectionsData):
    """Class representing a source of raw data for the intensity of
    the beam that is incident on the sample.

    :ivar label: Must be `'presample_intensity"`.
    :type label: Literal['presample_intensity']
    :ivar units: Must be `'counts'`.
    :type units: Literal['counts']
    """
    label: Literal['presample_intensity'] = 'presample_intensity'
    units: Literal['counts'] = 'counts'


class PostsampleIntensity(CorrectionsData):
    """Class representing a source of raw data for the intensity of
    the beam that has passed through the sample.

    :ivar label: Must be `'postsample_intensity'`.
    :type label: Literal['postsample_intensity']
    :ivar units: Must be `'counts'`.
    :type units: Literal['counts']
    """
    label: Literal['postsample_intensity'] = 'postsample_intensity'
    units: Literal['counts'] = 'counts'


class DwellTimeActual(CorrectionsData):
    """Class representing a source of raw data for the actual dwell
    time at each scan point in SPEC (with some scan types, this value
    can vary slightly point-to-point from the dwell time specified in
    the command).

    :ivar label: Must be `'dwell_time_actual'`.
    :type label: Literal['dwell_time_actual']
    :ivar units: Must be `'s'`.
    :type units: Literal['s']
    """
    label: Literal['dwell_time_actual'] = 'dwell_time_actual'
    units: Literal['s'] = 's'


class SpecConfig(CHAPBaseModel):
    """Class representing the raw data for one or more SPEC scans.

    :ivar station: The name of the station at which the data was
        collected.
    :type station: Literal['id1a3', 'id3a', 'id3b']
    :ivar spec_scans: A list of the SPEC scans that compose the set.
    :type spec_scans: list[SpecScans]
    :ivar experiment_type: Experiment type.
    :type experiment_type: Literal['EDD', 'GIWAXS', 'SAXSWAXS', 'TOMO',
        'XRF', 'HDRM']
    """
    station: Literal['id1a3', 'id3a', 'id3b', 'id4b']
    experiment_type: Literal[
        'EDD', 'GIWAXS', 'SAXSWAXS', 'TOMO', 'XRF', 'HDRM']
    spec_scans: conlist(item_type=SpecScans, min_length=1)

    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data):
        """Ensure that a valid configuration was provided and finalize
        spec_file filepaths.

        :param data: Pydantic validator data object.
        :type data: SpecConfig,
            pydantic_core._pydantic_core.ValidationInfo
        :return: The currently validated list of class properties.
        :rtype: dict
        """
        inputdir = data.get('inputdir')
        if inputdir is not None:
            spec_scans = data.get('spec_scans')
            for i, scans in enumerate(deepcopy(spec_scans)):
                if isinstance(scans, dict):
                    spec_file = scans['spec_file']
                    if not os.path.isabs(spec_file):
                        spec_scans[i]['spec_file'] = os.path.join(
                            inputdir, spec_file)
                else:
                    spec_file = scans.spec_file
                    if not os.path.isabs(spec_file):
                        spec_scans[i].spec_file = os.path.join(
                            inputdir, spec_file)
            data['spec_scans'] = spec_scans
        return data

    @field_validator('experiment_type')
    @classmethod
    def validate_experiment_type(cls, experiment_type, info):
        """Ensure values for the station and experiment_type fields are
        compatible.

        :param experiment_type: The value of `experiment_type` to
            validate.
        :type experiment_type: str
        :param info: Pydantic validator info object.
        :type info: pydantic_core._pydantic_core.ValidationInfo
        :raises ValueError: Invalid experiment type.
        :return: The validated field for `experiment_type`.
        :rtype: str
        """
        station = info.data.get('station')
        if station == 'id1a3':
            allowed_experiment_types = ['EDD', 'SAXSWAXS', 'TOMO']
        elif station == 'id3a':
            allowed_experiment_types = ['EDD', 'TOMO']
        elif station == 'id3b':
            allowed_experiment_types = ['GIWAXS', 'SAXSWAXS', 'TOMO', 'XRF']
        elif station == 'id4b':
            allowed_experiment_types = ['HDRM']
        else:
            allowed_experiment_types = []
        if experiment_type not in allowed_experiment_types:
            raise ValueError(
                f'For station {station}, allowed experiment types are '
                f'{", ".join(allowed_experiment_types)}. '
                f'Supplied experiment type {experiment_type} is not allowed.')
        import_scanparser(station, experiment_type)
        return experiment_type


class MapConfig(CHAPBaseModel):
    """Class representing an experiment consisting of one or more SPEC
    scans.

    :param: did: FOXDEN data identifier.
    :type did: str, optional
    :ivar title: The title for the map configuration.
    :type title: str
    :ivar station: The name of the station at which the map was
        collected.
    :type station: Literal['id1a3', 'id3a', 'id3b', id4b]
    :ivar experiment_type: Experiment type.
    :type experiment_type: Literal['EDD', 'GIWAXS', 'SAXSWAXS', 'TOMO',
        'XRF', 'HDRM']
    :ivar sample: The sample metadata configuration.
    :type sample: CHAP.commom.models.map.Sample
    :ivar spec_scans: A list of the SPEC scans that compose the map.
    :type spec_scans: list[SpecScans]
    :ivar scalar_data: A list of the sources of data representing
        other scalar raw data values collected at each point on the
        map. In the NeXus file representation of the map, datasets for
        these values will be included, defaults to `[]`.
    :type scalar_data: list[PointByPointScanData], optional
    :ivar independent_dimensions: A list of the sources of data
        representing the raw values of each independent dimension of
        the map.
    :type independent_dimensions: list[PointByPointScanData]
    :ivar presample_intensity: A source of point-by-point presample
        beam intensity data. Required when applying a CorrectionConfig
        tool.
    :type presample_intensity: PresampleIntensity, optional
    :ivar dwell_time_actual: A source of point-by-point actual dwell
        times for SPEC scans. Required when applying a
        CorrectionConfig tool.
    :type dwell_time_actual: DwellTimeActual, optional
    :ivar postsample_intensity: A source of point-by-point postsample
        beam intensity data. Required when applying a CorrectionConfig
        tool with `correction_type='flux_absorption'` or
        `correction_type='flux_absorption_background'`.
    :type postsample_intensity: PresampleIntensity, optional
    :ivar attrs: Additional Map configuration attributes.
    :type attrs: dict, optional
    """
    did: Optional[constr(strip_whitespace=True)] = None
    title: constr(strip_whitespace=True, min_length=1)
    station: Literal['id1a3', 'id3a', 'id3b', 'id4b']
    experiment_type: Literal[
        'EDD', 'GIWAXS', 'SAXSWAXS', 'TOMO', 'XRF', 'HDRM']
    sample: Sample
    spec_scans: conlist(item_type=SpecScans, min_length=1)
    scalar_data: Optional[conlist(item_type=PointByPointScanData)] = []
    independent_dimensions: conlist(
        item_type=IndependentDimension, min_length=1)
    presample_intensity: Optional[PresampleIntensity] = None
    dwell_time_actual: Optional[DwellTimeActual] = None
    postsample_intensity: Optional[PostsampleIntensity] = None
    attrs: Optional[Annotated[dict, Field(validate_default=True)]] = {}
#    _coords: dict = PrivateAttr()
    _dims: tuple = PrivateAttr()
#    _scan_step_indices: list = PrivateAttr()
#    _shape: tuple = PrivateAttr()

    _validate_independent_dimensions = field_validator(
        'independent_dimensions')(validate_data_source_for_map_config)
    _validate_presample_intensity = field_validator(
        'presample_intensity')(validate_data_source_for_map_config)
    _validate_dwell_time_actual = field_validator(
        'dwell_time_actual')(validate_data_source_for_map_config)
    _validate_postsample_intensity = field_validator(
        'postsample_intensity')(validate_data_source_for_map_config)
    _validate_scalar_data = field_validator(
        'scalar_data')(validate_data_source_for_map_config)

    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data):
        """Ensure that a valid configuration was provided and finalize
        spec_file filepaths.

        :param data: Pydantic validator data object.
        :type data:
            MapConfig, pydantic_core._pydantic_core.ValidationInfo
        :return: The currently validated list of class properties.
        :rtype: dict
        """
        if 'spec_file' in data and 'scan_numbers' in data:
            spec_file = data.pop('spec_file')
            scan_numbers = data.pop('scan_numbers')
            if 'spec_scans' in data:
                raise ValueError(
                    f'Ambiguous SPEC scan information: spec_file={spec_file},'
                    f' scan_numbers={scan_numbers}, and '
                    f'spec_scans={data["spec_scans"]}')
            data['spec_scans'] = [
                {'spec_file': spec_file, 'scan_numbers': scan_numbers}]
        else:
            inputdir = data.get('inputdir')
            if inputdir is not None:
                spec_scans = data.get('spec_scans')
                for i, scans in enumerate(deepcopy(spec_scans)):
                    spec_file = scans['spec_file']
                    if not os.path.isabs(spec_file):
                        spec_scans[i]['spec_file'] = os.path.join(
                            inputdir, spec_file)
                    spec_scans[i] = SpecScans(**spec_scans[i], **data)
                data['spec_scans'] = spec_scans
        return data

    @field_validator('experiment_type')
    @classmethod
    def validate_experiment_type(cls, experiment_type, info):
        """Ensure values for the station and experiment_type fields are
        compatible.

        :param experiment_type: The value of `experiment_type` to
            validate.
        :type experiment_type: dict
        :param info: Pydantic validator info object.
        :type info: pydantic_core._pydantic_core.ValidationInfo
        :raises ValueError: Invalid experiment type.
        :return: The validated field for `experiment_type`.
        :rtype: str
        """
        station = info.data['station']
        if station == 'id1a3':
            allowed_experiment_types = ['EDD', 'SAXSWAXS', 'TOMO']
        elif station == 'id3a':
            allowed_experiment_types = ['EDD', 'TOMO']
        elif station == 'id3b':
            allowed_experiment_types = ['GIWAXS', 'SAXSWAXS', 'TOMO', 'XRF']
        elif station == 'id4b':
            allowed_experiment_types = ['HDRM']
        else:
            allowed_experiment_types = []
        if experiment_type not in allowed_experiment_types:
            raise ValueError(
                f'For station {station}, allowed experiment types are '
                f'{", ".join(allowed_experiment_types)}. '
                f'Supplied experiment type {experiment_type} is not allowed.')
        return experiment_type

    #RV maybe better to use model_validator, see v2 docs?
    @field_validator('attrs')
    @classmethod
    def validate_attrs(cls, attrs, info):
        """Validate any additional attributes depending on the values
        for the station and experiment_type fields.

        :param attrs: Any additional attributes to the MapConfig class.
        :type attrs: dict
        :param info: Pydantic validator info object.
        :type info: pydantic_core._pydantic_core.ValidationInfo
        :raises ValueError: Invalid attribute.
        :return: The validated field for `attrs`.
        :rtype: dict
        """
        # Get the map's scan_type for EDD experiments
        values = info.data
        station = values['station']
        experiment_type = values['experiment_type']
        if station in ['id1a3', 'id3a'] and experiment_type == 'EDD':
            scan_type = cls.get_smb_par_attr(values, 'scan_type')
            if scan_type is not None:
                attrs['scan_type'] = scan_type
            attrs['config_id'] = cls.get_smb_par_attr(values, 'config_id')
            dataset_id = cls.get_smb_par_attr(values, 'dataset_id')
            if dataset_id is not None:
                attrs['dataset_id'] = dataset_id
            if attrs.get('scan_type') is None:
                return attrs
            axes_labels = {1: 'fly_labx', 2: 'fly_laby', 3: 'fly_labz',
                           4: 'fly_ometotal'}
            if attrs['scan_type'] != 0:
                attrs['fly_axis_labels'] = [
                    axes_labels[cls.get_smb_par_attr(values, 'fly_axis0')]]
            if attrs['scan_type'] in (2, 3, 5):
                attrs['fly_axis_labels'].append(
                    axes_labels[cls.get_smb_par_attr(values, 'fly_axis1')])
        return attrs

    @staticmethod
    def get_smb_par_attr(class_fields, label, units='-', name=None):
        """Read an SMB par file attribute.

        :param class_fields: The Map configuration class fields.
        :type class_fields: Any
        :param label: An attrs field key, the user-defined label for
            referring to this data in the NeXus file and in other
            tools.
        :type label: str
        :param units: The attrs' field unit, defaults to `'-'`.
        :type units: str
        :param name: The attrs' field name, the name with which these
            raw data were recorded at time of data collection,
            defaults to `label`.
        :type name: str, optional.
        """
        if name is None:
            name = label
        PointByPointScanData(
            label=label, data_type='smb_par', units=units, name=name)
        values = []
        for scans in class_fields.get('spec_scans'):
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                try:
                    values.append(scanparser.pars[name])
                except:
#                    print(
#                        f'Warning: No value found for .par file value "{name}"'
#                        f' on scan {scan_number} in spec file '
#                        f'{scans.spec_file}.')
                    values.append(None)
        values = list(set(values))
        if len(values) != 1:
            raise ValueError(f'More than one {name} in map not allowed '
                             f'({values})')
        return values[0]

    @property
    def all_scalar_data(self):
        """Return a list of all instances of `PointByPointScanData`
        for which this map configuration will collect dataset-like
        data (as opposed to axes-like data).

        This will be any and all of the items in the
        corrections-data-related fields, as well as any additional
        items in the optional `scalar_data` field.
        """
        return [getattr(self, label, None)
                for label in CorrectionsData.reserved_labels()
                if getattr(self, label, None) is not None] + self.scalar_data

    @property
    def coords(self):
        """Return a dictionary of the values of each independent
        dimension across the map.
        """
        raise RuntimeError('property coords not implemented')
        if not hasattr(self, '_coords'):
            fly_axis_labels = self.attrs.get('fly_axis_labels', [])
            coords = {}
            for dim in self.independent_dimensions:
                if dim.label in fly_axis_labels:
                    relative = True
                    ndigits = 3
                else:
                    relative = False
                    ndigits = None
                coords[dim.label] = []
                for scans in self.spec_scans:
                    for scan_number in scans.scan_numbers:
                        scanparser = scans.get_scanparser(scan_number)
                        for scan_step_index in range(
                                scanparser.spec_scan_npts):
                            coords[dim.label].append(dim.get_value(
                                    scans, scan_number, scan_step_index,
                                    self.scalar_data, relative, ndigits))
                if self.map_type == 'structured':
                    coords[dim.label] = np.unique(coords[dim.label])
            self._coords = coords
        return self._coords

    @property
    def dims(self):
        """Return a tuple of the independent dimension labels for the
        map.
        """
        if not hasattr(self, '_dims'):
            self._dims = [dim.label for dim in self.independent_dimensions]
        return self._dims

    @property
    def scan_step_indices(self):
        """Return an ordered list in which we can look up the SpecScans
        object, the scan number, and scan step index for every point
        on the map.
        """
        raise RuntimeError('property scan_step_indices not implemented')
        if not hasattr(self, '_scan_step_indices'):
            scan_step_indices = []
            for scans in self.spec_scans:
                for scan_number in scans.scan_numbers:
                    scanparser = scans.get_scanparser(scan_number)
                    for scan_step_index in range(scanparser.spec_scan_npts):
                        scan_step_indices.append(
                            (scans, scan_number, scan_step_index))
            self._scan_step_indices = scan_step_indices
        return self._scan_step_indices

    @property
    def shape(self):
        """Return the shape of the map -- a tuple representing the
        number of unique values of each dimension across the map.
        """
        raise RuntimeError('property shape not implemented')
        if not hasattr(self, '_shape'):
            if self.map_type == 'structured':
                self._shape = tuple([len(v) for k, v in self.coords.items()])
            else:
                self._shape =  (len(self.scan_step_indices),)
        return self._shape

    def get_coords(self, map_index):
        """Return a dictionary of the coordinate names and values of
        each independent dimension for a given point on the map.

        :param map_index: The map index to return coordinates for.
        :type map_index: tuple
        :return: A list of coordinate values.
        :rtype: dict
        """
        raise RuntimeError('get_coords not implemented')
        if self.map_type == 'structured':
            scan_type = self.attrs.get('scan_type', -1)
            fly_axis_labels = self.attrs.get('fly_axis_labels', [])
            if (scan_type in (3, 5)
                    and len(self.dims) ==
                        len(map_index) + len(fly_axis_labels)):
                dims = [dim for dim in self.dims if dim not in fly_axis_labels]
                return {dim:self.coords[dim][i]
                        for dim, i in zip(dims, map_index)}
            return {dim:self.coords[dim][i]
                    for dim, i in zip(self.dims, map_index)}
        return {dim:self.coords[dim][map_index[0]] for dim in self.dims}

    def get_detector_data(self, detector_name, map_index):
        """Return detector data collected by this map for a given
        point on the map.

        :param detector_name: Name of the detector for which to return
            data. Usually the value of the detector's EPICS
            areaDetector prefix macro, $P.
        :type detector_name: str
        :param map_index: The map index to return detector data for.
        :type map_index: tuple
        :return: One frame of raw detector data.
        :rtype: np.ndarray
        """
        raise RuntimeError('get_detector_data not implemented')
        scans, scan_number, scan_step_index = \
            self.get_scan_step_index(map_index)
        scanparser = scans.get_scanparser(scan_number)
        return scanparser.get_detector_data(detector_name, scan_step_index)

    def get_scan_step_index(self, map_index):
        """Return parameters to identify a single SPEC scan step that
        corresponds to the map point at the index provided.

        :param map_index: The index of a map point to identify as a
            specific SPEC scan step index.
        :type map_index: tuple
        :return: A `SpecScans` configuration, scan number, and scan
            step index.
        :rtype: tuple[SpecScans, int, int]
        """
        raise RuntimeError('get_scan_step_index not implemented')
        scan_type = self.attrs.get('scan_type', -1)
        fly_axis_labels = self.attrs.get('fly_axis_labels', [])
        if self.map_type == 'structured':
            map_coords = self.get_coords(map_index)
            for scans, scan_number, scan_step_index in self.scan_step_indices:
                coords = {dim.label:(
                              dim.get_value(
                                  scans, scan_number, scan_step_index,
                                  self.scalar_data, True, 3)
                              if dim.label in fly_axis_labels
                              else
                              dim.get_value(
                                  scans, scan_number, scan_step_index,
                                  self.scalar_data))
                          for dim in self.independent_dimensions}
                if coords == map_coords:
                    return scans, scan_number, scan_step_index
            raise RuntimeError(f'Unable to match coordinates {coords}')
        return self.scan_step_indices[map_index[0]]

    def get_value(self, data, map_index):
        """Return the raw data collected by a single device at a
        single point in the map.

        :param data: The device configuration to return a value of raw
            data for.
        :type data: PointByPointScanData
        :param map_index: The map index to return raw data for.
        :type map_index: tuple
        :return: Raw data value.
        """
        raise RuntimeError('get_value not implemented')
        scans, scan_number, scan_step_index = \
            self.get_scan_step_index(map_index)
        return data.get_value(scans, scan_number, scan_step_index,
                              self.scalar_data)


def import_scanparser(station, experiment):
    """Given the name of a CHESS station and experiment type, import
    the corresponding subclass of `ScanParser` as `ScanParser`.

    :param station: The station name
        ('IDxx', not the beamline acronym).
    :type station: str
    :param experiment: The experiment type.
    :type experiment: Literal[
        'EDD', 'GIWAXS', 'SAXSWAXS', 'TOMO', 'XRF', 'HDRM']
    """
    # Third party modules
    from chess_scanparsers import choose_scanparser

    globals()['ScanParser'] = choose_scanparser(station, experiment)
