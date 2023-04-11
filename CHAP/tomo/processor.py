#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
'''
File       : processor.py
Author     : Rolf Verberg <rolfverberg AT gmail dot com>
Description: Module for Processors used only by tomography experiments
'''

# system modules
from os import mkdir
from os import path as os_path
from time import time

# third party modules
from nexusformat.nexus import NXobject
import numpy as np

# local modules
from CHAP.common.utils.general import clear_plot, clear_imshow, quick_plot, quick_imshow
from CHAP.processor import Processor

num_core_tomopy_limit = 24


class TomoDataProcessor(Processor):
    '''Class representing the processes to reconstruct a set of Tomographic images returning
    either a dictionary or a `nexusformat.nexus.NXroot` object containing the (meta) data after
    processing each individual step.
    '''

    def _process(self, data):
        '''Process the output of a `Reader` that contains a map or a `nexusformat.nexus.NXroot`
        object and one that contains the step specific instructions and return either a dictionary
        or a `nexusformat.nexus.NXroot` returning the processed result.

        :param data: Result of `Reader.read` where at least one item is of type
            `nexusformat.nexus.NXroot` or has the value `'MapConfig'` for the `'schema'` key,
            and at least one item has the value `'TomoReduceConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :return: processed (meta)data
        :rtype: dict or nexusformat.nexus.NXroot
        '''

        tomo = Tomo(save_figs='only')
        nxroot = None
        center_config = None

        # Get and validate the relevant configuration objects in data
        configs = self.get_configs(data)

        # Setup the pipeline for a tomography reconstruction
        if 'setup' in configs:
            configs.pop('nxroot', None)
            nxroot = self.get_nxroot(configs.pop('map'), configs.pop('setup'))
        else:
            nxroot = configs.pop('nxroot', None)

        # Reduce tomography images
        if 'reduce' in configs:
            tool_config = configs.pop('reduce')
            if nxroot is None:
                map_config = configs.pop('map')
                nxroot = self.get_nxroot(map_config, tool_config)
            nxroot = tomo.gen_reduced_data(nxroot, img_x_bounds=tool_config.img_x_bounds)

        # Find rotation axis centers for the tomography stacks
        # Pass tool_config directly to tomo.find_centers?
        if 'find_center' in configs:
            tool_config = configs.pop('find_center')
            center_rows = [tool_config.lower_row, tool_config.upper_row]
            if (None in center_rows or tool_config.lower_center_offset is None or
                    tool_config.upper_center_offset is None):
                center_config = tomo.find_centers(nxroot, center_rows=center_rows,
                        center_stack_index=tool_config.center_stack_index)
            else:
                #RV make a convert to dict in basemodel?
                center_config = {'lower_row': tool_config.lower_row,
                        'lower_center_offset': tool_config.lower_center_offset,
                        'upper_row': tool_config.upper_row,
                        'upper_center_offset': tool_config.upper_center_offset}

        # Reconstruct tomography stacks
        # Pass tool_config and center_config directly to tomo.reconstruct_data
        if 'reconstruct' in configs:
            tool_config = configs.pop('reconstruct')
            nxroot = tomo.reconstruct_data(nxroot, center_config, x_bounds=tool_config.x_bounds,
                    y_bounds=tool_config.y_bounds, z_bounds=tool_config.z_bounds)
            center_config = None

        # Combine reconstructed tomography stacks
        if 'combine' in configs:
            tool_config = configs.pop('combine')
            nxroot = tomo.combine_data(nxroot, x_bounds=tool_config.x_bounds,
                    y_bounds=tool_config.y_bounds, z_bounds=tool_config.z_bounds)

        if center_config is not None:
            return center_config
        else:
            return nxroot

    def get_configs(self, data):
        '''Get instances of the configuration objects needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item
            is of type `nexusformat.nexus.NXroot` or has the value
            `'MapConfig'` for the `'schema'` key, and at least one item
            has the value `'TomoSetupConfig'`, or `'TomoReduceConfig'`,
            or `'TomoFindCenterConfig'`, or `'TomoReconstructConfig'`,
            or `'TomoCombineConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If valid config objects cannot be constructed
            from `data`.
        :return: valid instances of the configuration objects with field
            values taken from `data`.
        :rtype: dict
        '''
        #:rtype: dict{'map': MapConfig, 'reduce': TomoReduceConfig} RV: Is there a way to denote optional items?
        from CHAP.common.models import MapConfig
        from CHAP.tomo.models import TomoSetupConfig, TomoReduceConfig, TomoFindCenterConfig, \
                TomoReconstructConfig, TomoCombineConfig
        from nexusformat.nexus import NXroot

        configs = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    schema = item.get('schema')
                    if isinstance(item.get('data'), NXroot):
                        configs['nxroot'] = item.get('data')
                    if schema == 'MapConfig':
                        configs['map'] = MapConfig(**(item.get('data')))
                    if schema == 'TomoSetupConfig':
                        configs['setup'] = TomoSetupConfig(**(item.get('data')))
                    if schema == 'TomoReduceConfig':
                        configs['reduce'] = TomoReduceConfig(**(item.get('data')))
                    elif schema == 'TomoFindCenterConfig':
                        configs['find_center'] = TomoFindCenterConfig(**(item.get('data')))
                    elif schema == 'TomoReconstructConfig':
                        configs['reconstruct'] = TomoReconstructConfig(**(item.get('data')))
                    elif schema == 'TomoCombineConfig':
                        configs['combine'] = TomoCombineConfig(**(item.get('data')))

        return configs

    def get_nxroot(self, map_config, tool_config):
        '''Get a map of the collected tomography data from the scans in `map_config`. The
        data will be reduced based on additional parameters included in `tool_config`.
        The data will be returned along with relevant metadata in the form of a NeXus structure.

        :param map_config: the map configuration
        :type map_config: MapConfig
        :param tool_config: the tomography image reduction configuration
        :type tool_config: TomoReduceConfig
        :return: a map of the collected tomography data along with the data reduction configuration
        :rtype: nexusformat.nexus.NXroot
        '''
        from CHAP.common import MapProcessor
        from CHAP.common.models.map import import_scanparser
        from CHAP.common.utils.general import index_nearest
        from copy import deepcopy
        from nexusformat.nexus import NXcollection, NXdata, NXdetector, NXinstrument, NXsample, \
                NXsource, NXsubentry, NXroot

        include_raw_data = getattr(tool_config, "include_raw_data", False)

        # Construct NXroot
        nxroot = NXroot()

        # Construct base NXentry and add to NXroot
        nxentry = MapProcessor.get_nxentry(map_config)
        nxroot[map_config.title] = nxentry
        nxroot.attrs['default'] = map_config.title
        nxentry.definition = 'NXtomo'
        if 'data' in nxentry:
            del nxentry['data']

        # Add an NXinstrument to the NXentry
        nxinstrument = NXinstrument()
        nxentry.instrument = nxinstrument

        # Add an NXsource to the NXinstrument
        nxsource = NXsource()
        nxinstrument.source = nxsource
        nxsource.type = 'Synchrotron X-ray Source'
        nxsource.name = 'CHESS'
        nxsource.probe = 'x-ray'

        # Tag the NXsource with the runinfo (as an attribute)
#        nxsource.attrs['cycle'] = map_config.cycle
#        nxsource.attrs['btr'] = map_config.btr
        nxsource.attrs['station'] = map_config.station
        nxsource.attrs['experiment_type'] = map_config.experiment_type

        # Add an NXdetector to the NXinstrument (don't fill in data fields yet)
        nxdetector = NXdetector()
        nxinstrument.detector = nxdetector
        nxdetector.local_name = tool_config.detector.prefix
        pixel_size = tool_config.detector.pixel_size
        if len(pixel_size) == 1:
            nxdetector.x_pixel_size = pixel_size[0]/tool_config.detector.lens_magnification
            nxdetector.y_pixel_size = pixel_size[0]/tool_config.detector.lens_magnification
        else:
            nxdetector.x_pixel_size = pixel_size[0]/tool_config.detector.lens_magnification
            nxdetector.y_pixel_size = pixel_size[1]/tool_config.detector.lens_magnification
        nxdetector.x_pixel_size.attrs['units'] = 'mm'
        nxdetector.y_pixel_size.attrs['units'] = 'mm'

        if include_raw_data:
            # Add an NXsample to NXentry (don't fill in data fields yet)
            nxsample = NXsample()
            nxentry.sample = nxsample
            nxsample.name = map_config.sample.name
            nxsample.description = map_config.sample.description

        # Add NXcollection's to NXentry to hold metadata about the spec scans in the map
        # Also obtain the data fields in NXsample and NXdetector if requested
        import_scanparser(map_config.station, map_config.experiment_type)
        image_keys = []
        sequence_numbers = []
        image_stacks = []
        rotation_angles = []
        x_translations = []
        z_translations = []
        for scans in map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                if map_config.station in ('id1a3', 'id3a'):
                    scan_type = scanparser.scan_type
                    if scan_type == 'df1':
                        image_key = 2
                        field_name = 'dark_field'
                    elif scan_type == 'bf1':
                        image_key = 1
                        field_name = 'bright_field'
                    elif scan_type == 'ts1':
                        image_key = 0
                        field_name = 'tomo_fields'
                    else:
                        raise RuntimeError('Invalid scan type: {scan_type}')
                elif map_config.station in ('id3b'):
                    if scans.spec_file.endswith('_dark'):
                        image_key = 2
                        field_name = 'dark_field'
                    elif scans.spec_file.endswith('_flat'):
                        #RV not yet tested with an actual fmb run
                        image_key = 1
                        field_name = 'bright_field'
                    else:
                        image_key = 0
                        field_name = 'tomo_fields'
                else:
                    raise RuntimeError(f'Invalid station: {station}')

                # Create an NXcollection for each field type
                if field_name in nxentry.spec_scans:
                    nxcollection = nxentry.spec_scans[field_name]
                    if nxcollection.attrs['spec_file'] != str(scans.spec_file):
                        raise RuntimeError(f'Multiple SPEC files for a single field type not yet '+
                                f'implemented; field name: {field_name}, '+
                                f'SPEC file: {str(scans.spec_file)}')
                else:
                    nxcollection = NXcollection()
                    nxentry.spec_scans[field_name] = nxcollection
                    nxcollection.attrs['spec_file'] = str(scans.spec_file)
                    nxcollection.attrs['date'] = scanparser.spec_scan.file_date

                # Get thetas
                image_offset = scanparser.starting_image_offset
                if map_config.station in ('id1a3', 'id3a'):
                    theta_vals = scanparser.theta_vals
                    thetas = np.linspace(theta_vals.get('start'), theta_vals.get('end'),
                            theta_vals.get('num'))
                else:
                    if len(scans.scan_numbers) != 1:
                        raise RuntimeError('Multiple scans not yet implemented for '+
                                f'{map_config.station}')
                    scan_number = scans.scan_numbers[0]
                    thetas = []
                    for dim in map_config.independent_dimensions:
                        if dim.label != 'theta':
                            continue
                        for index in range(scanparser.spec_scan_npts):
                            thetas.append(dim.get_value(scans, scan_number, index))
                    if not len(thetas):
                        raise RuntimeError(f'Unable to obtain thetas for {field_name}')
                    if thetas[image_offset] <= 0.0 and thetas[-1] >= 180.0:
                        image_offset = index_nearest(thetas, 0.0)
                        thetas = thetas[image_offset:index_nearest(thetas, 180.0)]
                    elif thetas[-1]-thetas[image_offset] >= 180:
                        thetas = thetas[image_offset:index_nearest(thetas, thetas[0]+180.0)]
                    else:
                        thetas = thetas[image_offset:]

                # x and z translations
                x_translation = scanparser.horizontal_shift
                z_translation = scanparser.vertical_shift

                # Add an NXsubentry to the NXcollection for each scan
                entry_name = f'scan_{scan_number}'
                nxsubentry = NXsubentry()
                nxcollection[entry_name] = nxsubentry
                nxsubentry.start_time = scanparser.spec_scan.date
                nxsubentry.spec_command = scanparser.spec_command
                # Add an NXinstrument to the scan's NXsubentry
                nxsubentry.instrument = NXinstrument()
                # Add an NXdetector to the NXinstrument to the scan's NXsubentry
                nxsubentry.instrument.detector = deepcopy(nxdetector)
                nxsubentry.instrument.detector.frame_start_number = image_offset
                nxsubentry.instrument.detector.image_key = image_key
                # Add an NXsample to the scan's NXsubentry
                nxsubentry.sample = NXsample()
                nxsubentry.sample.rotation_angle = thetas
                nxsubentry.sample.rotation_angle.units = 'degrees'
                nxsubentry.sample.x_translation = x_translation
                nxsubentry.sample.x_translation.units = 'mm'
                nxsubentry.sample.z_translation = z_translation
                nxsubentry.sample.z_translation.units = 'mm'

                if include_raw_data:
                    num_image = len(thetas)
                    image_keys += num_image*[image_key]
                    sequence_numbers += list(range(num_image))
                    image_stacks.append(scanparser.get_detector_data(tool_config.detector.prefix,
                            scan_step_index=(image_offset, image_offset+num_image)))
                    rotation_angles += list(thetas)
                    x_translations += num_image*[x_translation]
                    z_translations += num_image*[z_translation]

        if include_raw_data:
            # Add image data to NXdetector
            nxinstrument.detector.image_key = image_keys
            nxinstrument.detector.sequence_number = sequence_numbers
            nxinstrument.detector.data = np.concatenate([image for image in image_stacks])

            # Add image data to NXsample
            nxsample.rotation_angle = rotation_angles
            nxsample.rotation_angle.attrs['units'] = 'degrees'
            nxsample.x_translation = x_translations
            nxsample.x_translation.attrs['units'] = 'mm'
            nxsample.z_translation = z_translations
            nxsample.z_translation.attrs['units'] = 'mm'

            # Add an NXdata to NXentry
            nxdata = NXdata()
            nxentry.data = nxdata
            nxdata.makelink(nxentry.instrument.detector.data, name='data')
            nxdata.makelink(nxentry.instrument.detector.image_key)
            nxdata.makelink(nxentry.sample.rotation_angle)
            nxdata.makelink(nxentry.sample.x_translation)
            nxdata.makelink(nxentry.sample.z_translation)
#            nxdata.attrs['axes'] = ['field', 'row', 'column']
#            nxdata.attrs['field_indices'] = 0
#            nxdata.attrs['row_indices'] = 1
#            nxdata.attrs['column_indices'] = 2

        return(nxroot)


def nxcopy(nxobject:NXobject, exclude_nxpaths:list[str]=[], nxpath_prefix:str='') -> NXobject:
    '''Function that returns a copy of a nexus object, optionally exluding certain child items.

    :param nxobject: the original nexus object to return a "copy" of
    :type nxobject: nexusformat.nexus.NXobject
    :param exlude_nxpaths: a list of paths to child nexus objects that
        should be exluded from the returned "copy", defaults to `[]`
    :type exclude_nxpaths: list[str], optional
    :param nxpath_prefix: For use in recursive calls from inside this
        function only!
    :type nxpath_prefix: str
    :return: a copy of `nxobject` with some children optionally exluded.
    :rtype: NXobject
    '''
    from nexusformat.nexus import NXgroup

    nxobject_copy = nxobject.__class__()
    if not len(nxpath_prefix):
        if 'default' in nxobject.attrs:
            nxobject_copy.attrs['default'] = nxobject.attrs['default']
    else:
        for k, v in nxobject.attrs.items():
            nxobject_copy.attrs[k] = v

    for k, v in nxobject.items():
        nxpath = os_path.join(nxpath_prefix, k)

        if nxpath in exclude_nxpaths:
            continue

        if isinstance(v, NXgroup):
            nxobject_copy[k] = nxcopy(v, exclude_nxpaths=exclude_nxpaths,
                    nxpath_prefix=os_path.join(nxpath_prefix, k))
        else:
            nxobject_copy[k] = v

    return(nxobject_copy)


class set_numexpr_threads:
    def __init__(self, num_core):
        from multiprocessing import cpu_count

        if num_core is None or num_core < 1 or num_core > cpu_count():
            self.num_core = cpu_count()
        else:
            self.num_core = num_core

    def __enter__(self):
        import numexpr as ne

        self.num_core_org = ne.set_num_threads(min(self.num_core, ne.MAX_THREADS))

    def __exit__(self, exc_type, exc_value, traceback):
        import numexpr as ne

        ne.set_num_threads(self.num_core_org)


class Tomo:
    """Processing tomography data with misalignment.
    """
    def __init__(self, galaxy_flag=False, num_core=-1, output_folder='.', save_figs=None,
            test_mode=False):
        """Initialize with optional config input file or dictionary
        """
        from logging import getLogger

        from multiprocessing import cpu_count

        self.__name__ = self.__class__.__name__
        self.logger = getLogger(self.__name__)
        self.logger.propagate = False

        if not isinstance(galaxy_flag, bool):
            raise ValueError(f'Invalid parameter galaxy_flag ({galaxy_flag})')
        self.galaxy_flag = galaxy_flag
        self.num_core = num_core
        if self.galaxy_flag:
            if output_folder != '.':
                self.logger.warning('Ignoring output_folder in galaxy mode')
            self.output_folder = '.'
            if test_mode != False:
                self.logger.warning('Ignoring test_mode in galaxy mode')
            self.test_mode = False
            if save_figs is not None:
                self.logger.warning('Ignoring save_figs in galaxy mode')
            save_figs = 'only'
        else:
            self.output_folder = os_path.abspath(output_folder)
            if not os_path.isdir(output_folder):
                mkdir(os_path.abspath(output_folder))
            if not isinstance(test_mode, bool):
                raise ValueError(f'Invalid parameter test_mode ({test_mode})')
            self.test_mode = test_mode
            if save_figs is None:
                save_figs = 'no'
        self.test_config = {}
        if self.test_mode:
            if save_figs != 'only':
                self.logger.warning('Ignoring save_figs in test mode')
            save_figs = 'only'
        if save_figs == 'only':
            self.save_only = True
            self.save_figs = True
        elif save_figs == 'yes':
            self.save_only = False
            self.save_figs = True
        elif save_figs == 'no':
            self.save_only = False
            self.save_figs = False
        else:
            raise ValueError(f'Invalid parameter save_figs ({save_figs})')
        if self.save_only:
            self.block = False
        else:
            self.block = True
        if self.num_core == -1:
            self.num_core = cpu_count()
        if not isinstance(self.num_core, int) or self.num_core < 0:
            raise ValueError(f'Invalid parameter num_core ({num_core})')
        if self.num_core > cpu_count():
            self.logger.warning(f'num_core = {self.num_core} is larger than the number of '
                    f'available processors and reduced to {cpu_count()}')
            self.num_core= cpu_count()

    def read(self, filename):
        extension = os_path.splitext(filename)[1]
        if extension == '.yml' or extension == '.yaml':
            with open(filename, 'r') as f:
                config = safe_load(f)
#            if len(config) > 1:
#                raise ValueError(f'Multiple root entries in {filename} not yet implemented')
#            if len(list(config.values())[0]) > 1:
#                raise ValueError(f'Multiple sample maps in {filename} not yet implemented')
            return(config)
        elif extension == '.nxs':
            with NXFile(filename, mode='r') as nxfile:
                nxroot = nxfile.readfile()
            return(nxroot)
        else:
            raise ValueError(f'Invalid filename extension ({extension})')

    def write(self, data, filename):
        extension = os_path.splitext(filename)[1]
        if extension == '.yml' or extension == '.yaml':
            with open(filename, 'w') as f:
                safe_dump(data, f)
        elif extension == '.nxs':
            data.save(filename, mode='w')
        elif extension == '.nc':
            data.to_netcdf(os_path=filename)
        else:
            raise ValueError(f'Invalid filename extension ({extension})')

    def gen_reduced_data(self, data, img_x_bounds=None):
        """Generate the reduced tomography images.
        """
        from nexusformat.nexus import NXdata, NXprocess, NXroot

        from CHAP.common.models.map import import_scanparser

        self.logger.info('Generate the reduced tomography images')
        if img_x_bounds is not None:
            if not isinstance(img_x_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter img_x_bounds ({img_x_bounds})')
            img_x_bounds = tuple(img_x_bounds)

        # Create plot galaxy path directory if needed
        if self.galaxy_flag and not os_path.exists('tomo_reduce_plots'):
            mkdir('tomo_reduce_plots')

        if isinstance(data, dict):
            # Create Nexus format object from input dictionary
            wf = TomoWorkflow(**data)
            if len(wf.sample_maps) > 1:
                raise ValueError(f'Multiple sample maps not yet implemented')
            nxroot = NXroot()
            t0 = time()
            for sample_map in wf.sample_maps:
                self.logger.info(f'Start constructing the {sample_map.title} map.')
                import_scanparser(sample_map.station)
                sample_map.construct_nxentry(nxroot, include_raw_data=False)
            self.logger.info(f'Constructed all sample maps in {time()-t0:.2f} seconds.')
            nxentry = nxroot[nxroot.attrs['default']]
            # Get test mode configuration info
            if self.test_mode:
                self.test_config = data['sample_maps'][0]['test_mode']
        elif isinstance(data, NXroot):
            nxentry = data[data.attrs['default']]
        else:
            raise ValueError(f'Invalid parameter data ({data})')
        if 'data' in nxentry:
            del nxentry['data']

        # Create an NXprocess to store data reduction (meta)data
        reduced_data = NXprocess()

        # Generate dark field
        if 'dark_field' in nxentry['spec_scans']:
            reduced_data = self._gen_dark(nxentry, reduced_data)

        # Generate bright field
        reduced_data = self._gen_bright(nxentry, reduced_data)

        # Set vertical detector bounds for image stack
        img_x_bounds = self._set_detector_bounds(nxentry, reduced_data, img_x_bounds=img_x_bounds)
        self.logger.info(f'img_x_bounds = {img_x_bounds}')
        reduced_data['img_x_bounds'] = img_x_bounds

        # Set zoom and/or theta skip to reduce memory the requirement
        zoom_perc, num_theta_skip = self._set_zoom_or_skip()
        if zoom_perc is not None:
            reduced_data.attrs['zoom_perc'] = zoom_perc
        if num_theta_skip is not None:
            reduced_data.attrs['num_theta_skip'] = num_theta_skip

        # Generate reduced tomography fields
        reduced_data = self._gen_tomo(nxentry, reduced_data)

        # Create a copy of the input Nexus object and remove raw and any existing reduced data
        if isinstance(data, NXroot):
            exclude_items = [f'{nxentry._name}/reduced_data/data',
                    f'{nxentry._name}/instrument/detector/data',
                    f'{nxentry._name}/instrument/detector/image_key',
                    f'{nxentry._name}/instrument/detector/sequence_number',
                    f'{nxentry._name}/sample/rotation_angle',
                    f'{nxentry._name}/sample/x_translation',
                    f'{nxentry._name}/sample/z_translation',
                    f'{nxentry._name}/data/data',
                    f'{nxentry._name}/data/image_key',
                    f'{nxentry._name}/data/rotation_angle',
                    f'{nxentry._name}/data/x_translation',
                    f'{nxentry._name}/data/z_translation']
            nxroot = nxcopy(data, exclude_nxpaths=exclude_items)
            nxentry = nxroot[nxroot.attrs['default']]

        # Add the reduced data NXprocess
        nxentry.reduced_data = reduced_data

        if 'data' not in nxentry:
            nxentry.data = NXdata()
        nxentry.attrs['default'] = 'data'
        nxentry.data.makelink(nxentry.reduced_data.data.tomo_fields, name='reduced_data')
        nxentry.data.makelink(nxentry.reduced_data.rotation_angle, name='rotation_angle')
        nxentry.data.attrs['signal'] = 'reduced_data'
 
        return(nxroot)

    def find_centers(self, nxroot, center_rows=None, center_stack_index=None):
        """Find the calibrated center axis info
        """
        from nexusformat.nexus import NXentry, NXroot

        from CHAP.common.utils.general import is_int_pair

        self.logger.info('Find the calibrated center axis info')

        if not isinstance(nxroot, NXroot):
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        nxentry = nxroot[nxroot.attrs['default']]
        if not isinstance(nxentry, NXentry):
            raise ValueError(f'Invalid nxentry ({nxentry})')
        if self.galaxy_flag:
            if center_rows is not None:
                center_rows = tuple(center_rows)
                if not is_int_pair(center_rows):
                    raise ValueError(f'Invalid parameter center_rows ({center_rows})')
        elif center_rows is not None:
#            self.logger.warning(f'Ignoring parameter center_rows ({center_rows})')
#            center_rows = None
             if not isinstance(center_rows, (tuple, list)) or len(center_rows) != 2:
                raise ValueError(f'Invalid parameter center_rows ({center_rows})')
        if self.galaxy_flag:
            if center_stack_index is not None and (not isinstance(center_stack_index, int) or
                    center_stack_index < 0):
                raise ValueError(f'Invalid parameter center_stack_index ({center_stack_index})')

        # Create plot galaxy path directory and path if needed
        if self.galaxy_flag:
            if not os_path.exists('tomo_find_centers_plots'):
                mkdir('tomo_find_centers_plots')
            path = 'tomo_find_centers_plots'
        else:
            path = self.output_folder

        # Check if reduced data is available
        if ('reduced_data' not in nxentry or 'reduced_data' not in nxentry.data):
            raise KeyError(f'Unable to find valid reduced data in {nxentry}.')

        # Select the image stack to calibrate the center axis
        #   reduced data axes order: stack,theta,row,column
        #   Note: Nexus cannot follow a link if the data it points to is too big,
        #         so get the data from the actual place, not from nxentry.data
        tomo_fields_shape = nxentry.reduced_data.data.tomo_fields.shape
        if len(tomo_fields_shape) != 4 or any(True for dim in tomo_fields_shape if not dim):
            raise KeyError('Unable to load the required reduced tomography stack')
        num_tomo_stacks = tomo_fields_shape[0]
        if num_tomo_stacks == 1:
            center_stack_index = 0
            default = 'n'
        else:
            if self.test_mode:
                center_stack_index = self.test_config['center_stack_index']-1 # make offset 0
            elif self.galaxy_flag:
                if center_stack_index is None:
                    center_stack_index = int(num_tomo_stacks/2)
                if center_stack_index >= num_tomo_stacks:
                    raise ValueError(f'Invalid parameter center_stack_index ({center_stack_index})')
            else:
                if center_stack_index is None:
                    center_stack_index = input_int('\nEnter tomography stack index to calibrate '
                            'the center axis', ge=1, le=num_tomo_stacks,
                            default=int(1+num_tomo_stacks/2))
                else:
                    if (not isinstance(center_stack_index, int) or
                            not 0 < center_stack_index <= num_tomo_stacks):
                        raise ValueError('Invalid parameter center_stack_index '+
                                f'({center_stack_index})')
                center_stack_index -= 1
            default = 'y'

        # Get thetas (in degrees)
        thetas = np.asarray(nxentry.reduced_data.rotation_angle)

        # Get effective pixel_size
        if 'zoom_perc' in nxentry.reduced_data:
            eff_pixel_size = 100.*(nxentry.instrument.detector.x_pixel_size/
                nxentry.reduced_data.attrs['zoom_perc'])
        else:
            eff_pixel_size = nxentry.instrument.detector.x_pixel_size

        # Get cross sectional diameter
        cross_sectional_dim = tomo_fields_shape[3]*eff_pixel_size
        self.logger.debug(f'cross_sectional_dim = {cross_sectional_dim}')

        # Determine center offset at sample row boundaries
        self.logger.info('Determine center offset at sample row boundaries')

        # Lower row center
        if self.test_mode:
            lower_row = self.test_config['lower_row']
        elif self.galaxy_flag:
            if center_rows is None:
                lower_row = 0
            else:
                lower_row = min(center_rows)
                if not 0 <= lower_row < tomo_fields_shape[2]-1:
                    raise ValueError(f'Invalid parameter center_rows ({center_rows})')
        else:
            if center_rows is not None and center_rows[0] is not None:
                lower_row = center_rows[0]
                if lower_row == -1:
                    lower_row = 0
                if not 0 <= lower_row < tomo_fields_shape[2]-1:
                    raise ValueError(f'Invalid parameter center_rows ({center_rows})')
            else:
                lower_row = select_one_image_bound(
                        nxentry.reduced_data.data.tomo_fields[center_stack_index,0,:,:],
                        0, bound=0, title=f'theta={round(thetas[0], 2)+0}',
                        bound_name='row index to find lower center', default=default,
                        raise_error=True)
        self.logger.debug('Finding center...')
        t0 = time()
        lower_center_offset = self._find_center_one_plane(
                #np.asarray(nxentry.reduced_data.data.tomo_fields[center_stack_index,:,lower_row,:]),
                nxentry.reduced_data.data.tomo_fields[center_stack_index,:,lower_row,:],
                lower_row, thetas, eff_pixel_size, cross_sectional_dim, path=path,
                num_core=self.num_core)
        self.logger.debug(f'... done in {time()-t0:.2f} seconds')
        self.logger.debug(f'lower_row = {lower_row:.2f}')
        self.logger.debug(f'lower_center_offset = {lower_center_offset:.2f}')

        # Upper row center
        if self.test_mode:
            upper_row = self.test_config['upper_row']
        elif self.galaxy_flag:
            if center_rows is None:
                upper_row = tomo_fields_shape[2]-1
            else:
                upper_row = max(center_rows)
                if not lower_row < upper_row < tomo_fields_shape[2]:
                    raise ValueError(f'Invalid parameter center_rows ({center_rows})')
        else:
            if center_rows is not None and center_rows[1] is not None:
                upper_row = center_rows[1]
                if upper_row == -1:
                    upper_row = tomo_fields_shape[2]-1
                if not lower_row < upper_row < tomo_fields_shape[2]:
                    raise ValueError(f'Invalid parameter center_rows ({center_rows})')
            else:
                upper_row = select_one_image_bound(
                        nxentry.reduced_data.data.tomo_fields[center_stack_index,0,:,:],
                        0, bound=tomo_fields_shape[2]-1, title=f'theta={round(thetas[0], 2)+0}',
                        bound_name='row index to find upper center', default=default,
                        raise_error=True)
        self.logger.debug('Finding center...')
        t0 = time()
        upper_center_offset = self._find_center_one_plane(
                #np.asarray(nxentry.reduced_data.data.tomo_fields[center_stack_index,:,upper_row,:]),
                nxentry.reduced_data.data.tomo_fields[center_stack_index,:,upper_row,:],
                upper_row, thetas, eff_pixel_size, cross_sectional_dim, path=path,
                num_core=self.num_core)
        self.logger.debug(f'... done in {time()-t0:.2f} seconds')
        self.logger.debug(f'upper_row = {upper_row:.2f}')
        self.logger.debug(f'upper_center_offset = {upper_center_offset:.2f}')

        center_config = {'lower_row': lower_row, 'lower_center_offset': lower_center_offset,
                'upper_row': upper_row, 'upper_center_offset': upper_center_offset}
        if num_tomo_stacks > 1:
            center_config['center_stack_index'] = center_stack_index+1 # save as offset 1

        # Save test data to file
        if self.test_mode:
            with open(f'{self.output_folder}/center_config.yaml', 'w') as f:
                safe_dump(center_config, f)

        return(center_config)

    def reconstruct_data(self, nxroot, center_info, x_bounds=None, y_bounds=None, z_bounds=None):
        """Reconstruct the tomography data.
        """
        from nexusformat.nexus import NXdata, NXentry, NXprocess, NXroot

        from CHAP.common.utils.general import is_int_pair

        self.logger.info('Reconstruct the tomography data')

        if not isinstance(nxroot, NXroot):
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        nxentry = nxroot[nxroot.attrs['default']]
        if not isinstance(nxentry, NXentry):
            raise ValueError(f'Invalid nxentry ({nxentry})')
        if not isinstance(center_info, dict):
            raise ValueError(f'Invalid parameter center_info ({center_info})')
        if x_bounds is not None:
            if not isinstance(x_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
            x_bounds = tuple(x_bounds)
        if y_bounds is not None:
            if not isinstance(y_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
            y_bounds = tuple(y_bounds)
        if z_bounds is not None:
            if not isinstance(z_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter z_bounds ({z_bounds})')
            z_bounds = tuple(z_bounds)

        # Create plot galaxy path directory and path if needed
        if self.galaxy_flag:
            if not os_path.exists('tomo_reconstruct_plots'):
                mkdir('tomo_reconstruct_plots')
            path = 'tomo_reconstruct_plots'
        else:
            path = self.output_folder

        # Check if reduced data is available
        if ('reduced_data' not in nxentry or 'reduced_data' not in nxentry.data):
            raise KeyError(f'Unable to find valid reduced data in {nxentry}.')

        # Create an NXprocess to store image reconstruction (meta)data
        nxprocess = NXprocess()

        # Get rotation axis rows and centers
        lower_row = center_info.get('lower_row')
        lower_center_offset = center_info.get('lower_center_offset')
        upper_row = center_info.get('upper_row')
        upper_center_offset = center_info.get('upper_center_offset')
        if (lower_row is None or lower_center_offset is None or upper_row is None or
                upper_center_offset is None):
            raise KeyError(f'Unable to find valid calibrated center axis info in {center_info}.')
        center_slope = (upper_center_offset-lower_center_offset)/(upper_row-lower_row)

        # Get thetas (in degrees)
        thetas = np.asarray(nxentry.reduced_data.rotation_angle)

        # Reconstruct tomography data
        #   reduced data axes order: stack,theta,row,column
        #   reconstructed data order in each stack: row/z,x,y
        #   Note: Nexus cannot follow a link if the data it points to is too big,
        #         so get the data from the actual place, not from nxentry.data
        if 'zoom_perc' in nxentry.reduced_data:
            res_title = f'{nxentry.reduced_data.attrs["zoom_perc"]}p'
        else:
            res_title = 'fullres'
        load_error = False
        num_tomo_stacks = nxentry.reduced_data.data.tomo_fields.shape[0]
        tomo_recon_stacks = num_tomo_stacks*[np.array([])]
        for i in range(num_tomo_stacks):
            # Convert reduced data stack from theta,row,column to row,theta,column
            self.logger.debug(f'Reading reduced data stack {i+1}...')
            t0 = time()
            tomo_stack = np.asarray(nxentry.reduced_data.data.tomo_fields[i])
            self.logger.debug(f'... done in {time()-t0:.2f} seconds')
            if len(tomo_stack.shape) != 3 or any(True for dim in tomo_stack.shape if not dim):
                raise ValueError(f'Unable to load tomography stack {i+1} for reconstruction')
            tomo_stack = np.swapaxes(tomo_stack, 0, 1)
            assert(len(thetas) == tomo_stack.shape[1])
            assert(0 <= lower_row < upper_row < tomo_stack.shape[0])
            center_offsets = [lower_center_offset-lower_row*center_slope,
                    upper_center_offset+(tomo_stack.shape[0]-1-upper_row)*center_slope]
            t0 = time()
            self.logger.debug(f'Running _reconstruct_one_tomo_stack on {self.num_core} cores ...')
            tomo_recon_stack = self._reconstruct_one_tomo_stack(tomo_stack, thetas,
                    center_offsets=center_offsets, num_core=self.num_core, algorithm='gridrec')
            self.logger.debug(f'... done in {time()-t0:.2f} seconds')
            self.logger.info(f'Reconstruction of stack {i+1} took {time()-t0:.2f} seconds')

            # Combine stacks
            tomo_recon_stacks[i] = tomo_recon_stack

        # Resize the reconstructed tomography data
        #   reconstructed data order in each stack: row/z,x,y
        if self.test_mode:
            x_bounds = tuple(self.test_config.get('x_bounds'))
            y_bounds = tuple(self.test_config.get('y_bounds'))
            z_bounds = None
        elif self.galaxy_flag:
            if x_bounds is not None and not is_int_pair(x_bounds, ge=0,
                    lt=tomo_recon_stacks[0].shape[1]):
                raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
            if y_bounds is not None and not is_int_pair(y_bounds, ge=0,
                    lt=tomo_recon_stacks[0].shape[1]):
                raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
            z_bounds = None
        else:
            x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(tomo_recon_stacks,
                   x_bounds=x_bounds, y_bounds=y_bounds, z_bounds=z_bounds)
        if x_bounds is None:
            x_range = (0, tomo_recon_stacks[0].shape[1])
            x_slice = int(x_range[1]/2)
        else:
            x_range = (min(x_bounds), max(x_bounds))
            x_slice = int((x_bounds[0]+x_bounds[1])/2)
        if y_bounds is None:
            y_range = (0, tomo_recon_stacks[0].shape[2])
            y_slice = int(y_range[1]/2)
        else:
            y_range = (min(y_bounds), max(y_bounds))
            y_slice = int((y_bounds[0]+y_bounds[1])/2)
        if z_bounds is None:
            z_range = (0, tomo_recon_stacks[0].shape[0])
            z_slice = int(z_range[1]/2)
        else:
            z_range = (min(z_bounds), max(z_bounds))
            z_slice = int((z_bounds[0]+z_bounds[1])/2)

        # Plot a few reconstructed image slices
        if self.save_figs:
            for i, stack in enumerate(tomo_recon_stacks):
                if num_tomo_stacks == 1:
                    basetitle = 'recon'
                else:
                    basetitle = f'recon stack {i+1}'
                title = f'{basetitle} {res_title} xslice{x_slice}'
                quick_imshow(stack[z_range[0]:z_range[1],x_slice,y_range[0]:y_range[1]],
                        title=title, path=path, save_fig=True, save_only=True)
                title = f'{basetitle} {res_title} yslice{y_slice}'
                quick_imshow(stack[z_range[0]:z_range[1],x_range[0]:x_range[1],y_slice],
                        title=title, path=path, save_fig=True, save_only=True)
                title = f'{basetitle} {res_title} zslice{z_slice}'
                quick_imshow(stack[z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]],
                        title=title, path=path, save_fig=True, save_only=True)

        # Save test data to file
        #   reconstructed data order in each stack: row/z,x,y
        if self.test_mode:
            for i, stack in enumerate(tomo_recon_stacks):
                np.savetxt(f'{self.output_folder}/recon_stack_{i+1}.txt',
                        stack[z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]], fmt='%.6e')

        # Add image reconstruction to reconstructed data NXprocess
        #   reconstructed data order in each stack: row/z,x,y
        nxprocess.data = NXdata()
        nxprocess.attrs['default'] = 'data'
        for k, v in center_info.items():
            nxprocess[k] = v
        if x_bounds is not None:
            nxprocess.x_bounds = x_bounds
        if y_bounds is not None:
            nxprocess.y_bounds = y_bounds
        if z_bounds is not None:
            nxprocess.z_bounds = z_bounds
        nxprocess.data['reconstructed_data'] = np.asarray([stack[z_range[0]:z_range[1],
                x_range[0]:x_range[1],y_range[0]:y_range[1]] for stack in tomo_recon_stacks])
        nxprocess.data.attrs['signal'] = 'reconstructed_data'

        # Create a copy of the input Nexus object and remove reduced data
        exclude_items = [f'{nxentry._name}/reduced_data/data', f'{nxentry._name}/data/reduced_data']
        nxroot_copy = nxcopy(nxroot, exclude_nxpaths=exclude_items)

        # Add the reconstructed data NXprocess to the new Nexus object
        nxentry_copy = nxroot_copy[nxroot_copy.attrs['default']]
        nxentry_copy.reconstructed_data = nxprocess
        if 'data' not in nxentry_copy:
            nxentry_copy.data = NXdata()
        nxentry_copy.attrs['default'] = 'data'
        nxentry_copy.data.makelink(nxprocess.data.reconstructed_data, name='reconstructed_data')
        nxentry_copy.data.attrs['signal'] = 'reconstructed_data'

        return(nxroot_copy)

    def combine_data(self, nxroot, x_bounds=None, y_bounds=None, z_bounds=None):
        """Combine the reconstructed tomography stacks.
        """
        from nexusformat.nexus import NXdata, NXentry, NXprocess, NXroot

        from CHAP.common.utils.general import is_int_pair

        self.logger.info('Combine the reconstructed tomography stacks')

        if not isinstance(nxroot, NXroot):
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        nxentry = nxroot[nxroot.attrs['default']]
        if not isinstance(nxentry, NXentry):
            raise ValueError(f'Invalid nxentry ({nxentry})')
        if x_bounds is not None:
            if not isinstance(x_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
            x_bounds = tuple(x_bounds)
        if y_bounds is not None:
            if not isinstance(y_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
            y_bounds = tuple(y_bounds)
        if z_bounds is not None:
            if not isinstance(z_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter z_bounds ({z_bounds})')
            z_bounds = tuple(z_bounds)

        # Create plot galaxy path directory and path if needed
        if self.galaxy_flag:
            if not os_path.exists('tomo_combine_plots'):
                mkdir('tomo_combine_plots')
            path = 'tomo_combine_plots'
        else:
            path = self.output_folder

        # Check if reconstructed image data is available
        if ('reconstructed_data' not in nxentry or 'reconstructed_data' not in nxentry.data):
            raise KeyError(f'Unable to find valid reconstructed image data in {nxentry}.')

        # Create an NXprocess to store combined image reconstruction (meta)data
        nxprocess = NXprocess()

        # Get the reconstructed data
        #   reconstructed data order: stack,row(z),x,y
        #   Note: Nexus cannot follow a link if the data it points to is too big,
        #         so get the data from the actual place, not from nxentry.data
        num_tomo_stacks = nxentry.reconstructed_data.data.reconstructed_data.shape[0]
        if num_tomo_stacks == 1:
            self.logger.info('Only one stack available: leaving combine_data')
            return(None)

        # Combine the reconstructed stacks
        # (load one stack at a time to reduce risk of hitting Nexus data access limit)
        t0 = time()
        self.logger.debug(f'Combining the reconstructed stacks ...')
        tomo_recon_combined = np.asarray(nxentry.reconstructed_data.data.reconstructed_data[0])
        if num_tomo_stacks > 2:
            tomo_recon_combined = np.concatenate([tomo_recon_combined]+
                    [nxentry.reconstructed_data.data.reconstructed_data[i]
                    for i in range(1, num_tomo_stacks-1)])
        if num_tomo_stacks > 1:
            tomo_recon_combined = np.concatenate([tomo_recon_combined]+
                    [nxentry.reconstructed_data.data.reconstructed_data[num_tomo_stacks-1]])
        self.logger.debug(f'... done in {time()-t0:.2f} seconds')
        self.logger.info(f'Combining the reconstructed stacks took {time()-t0:.2f} seconds')

        # Resize the combined tomography data stacks
        #   combined data order: row/z,x,y
        if self.test_mode:
            x_bounds = None
            y_bounds = None
            z_bounds = tuple(self.test_config.get('z_bounds'))
        elif self.galaxy_flag:
            if x_bounds is not None and not is_int_pair(x_bounds, ge=0,
                    lt=tomo_recon_stacks[0].shape[1]):
                raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
            if y_bounds is not None and not is_int_pair(y_bounds, ge=0,
                    lt=tomo_recon_stacks[0].shape[1]):
                raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
            z_bounds = None
        else:
            if x_bounds is None and x_bounds in nxentry.reconstructed_data:
                x_bounds = (-1, -1)
            if y_bounds is None and y_bounds in nxentry.reconstructed_data:
                y_bounds = (-1, -1)
            x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(tomo_recon_combined,
                    z_only=True)
        if x_bounds is None:
            x_range = (0, tomo_recon_combined.shape[1])
            x_slice = int(x_range[1]/2)
        else:
            x_range = x_bounds
            x_slice = int((x_bounds[0]+x_bounds[1])/2)
        if y_bounds is None:
            y_range = (0, tomo_recon_combined.shape[2])
            y_slice = int(y_range[1]/2)
        else:
            y_range = y_bounds
            y_slice = int((y_bounds[0]+y_bounds[1])/2)
        if z_bounds is None:
            z_range = (0, tomo_recon_combined.shape[0])
            z_slice = int(z_range[1]/2)
        else:
            z_range = z_bounds
            z_slice = int((z_bounds[0]+z_bounds[1])/2)

        # Plot a few combined image slices
        if self.save_figs:
            quick_imshow(tomo_recon_combined[z_range[0]:z_range[1],x_slice,y_range[0]:y_range[1]],
                    title=f'recon combined xslice{x_slice}', path=path, save_fig=True,
                    save_only=True)
            quick_imshow(tomo_recon_combined[z_range[0]:z_range[1],x_range[0]:x_range[1],y_slice],
                    title=f'recon combined yslice{y_slice}', path=path, save_fig=True,
                    save_only=True)
            quick_imshow(tomo_recon_combined[z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]],
                    title=f'recon combined zslice{z_slice}', path=path, save_fig=True,
                    save_only=True)

        # Save test data to file
        #   combined data order: row/z,x,y
        if self.test_mode:
            np.savetxt(f'{self.output_folder}/recon_combined.txt', tomo_recon_combined[
                    z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]], fmt='%.6e')

        # Add image reconstruction to reconstructed data NXprocess
        #   combined data order: row/z,x,y
        nxprocess.data = NXdata()
        nxprocess.attrs['default'] = 'data'
        if x_bounds is not None:
            nxprocess.x_bounds = x_bounds
        if y_bounds is not None:
            nxprocess.y_bounds = y_bounds
        if z_bounds is not None:
            nxprocess.z_bounds = z_bounds
        nxprocess.data['combined_data'] = tomo_recon_combined[
                z_range[0]:z_range[1],x_range[0]:x_range[1],y_range[0]:y_range[1]]
        nxprocess.data.attrs['signal'] = 'combined_data'

        # Create a copy of the input Nexus object and remove reconstructed data
        exclude_items = [f'{nxentry._name}/reconstructed_data/data',
                f'{nxentry._name}/data/reconstructed_data']
        nxroot_copy = nxcopy(nxroot, exclude_nxpaths=exclude_items)

        # Add the combined data NXprocess to the new Nexus object
        nxentry_copy = nxroot_copy[nxroot_copy.attrs['default']]
        nxentry_copy.combined_data = nxprocess
        if 'data' not in nxentry_copy:
            nxentry_copy.data = NXdata()
        nxentry_copy.attrs['default'] = 'data'
        nxentry_copy.data.makelink(nxprocess.data.combined_data, name='combined_data')
        nxentry_copy.data.attrs['signal'] = 'combined_data'

        return(nxroot_copy)

    def _gen_dark(self, nxentry, reduced_data):
        """Generate dark field.
        """
        from nexusformat.nexus import NXdata

        from CHAP.common.models.map import get_scanparser, import_scanparser

        # Get the dark field images
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key and 'data' in nxentry.instrument.detector:
            field_indices = [index for index, key in enumerate(image_key) if key == 2]
            tdf_stack = nxentry.instrument.detector.data[field_indices,:,:]
            # RV the default NXtomo form does not accomodate bright or dark field stacks
        else:
            import_scanparser(nxentry.instrument.source.attrs['station'],
                    nxentry.instrument.source.attrs['experiment_type'])
            dark_field_scans = nxentry.spec_scans.dark_field
            detector_prefix = str(nxentry.instrument.detector.local_name)
            tdf_stack = []
            for nxsubentry_name, nxsubentry in dark_field_scans.items():
                scan_number = int(nxsubentry_name.split('_')[-1])
                scanparser = get_scanparser(dark_field_scans.attrs['spec_file'], scan_number)
                image_offset = int(nxsubentry.instrument.detector.frame_start_number)
                num_image = len(nxsubentry.sample.rotation_angle)
                tdf_stack.append(scanparser.get_detector_data(detector_prefix,
                        (image_offset, image_offset+num_image)))
            if isinstance(tdf_stack, list):
                assert(len(tdf_stack) == 1) # TODO
                tdf_stack = tdf_stack[0]

        # Take median
        if tdf_stack.ndim == 2:
            tdf = tdf_stack
        elif tdf_stack.ndim == 3:
            tdf = np.median(tdf_stack, axis=0)
            del tdf_stack
        else:
           raise ValueError(f'Invalid tdf_stack shape ({tdf_stack.shape})')

        # Remove dark field intensities above the cutoff
#RV        tdf_cutoff = None
        tdf_cutoff = tdf.min()+2*(np.median(tdf)-tdf.min())
        self.logger.debug(f'tdf_cutoff = {tdf_cutoff}')
        if tdf_cutoff is not None:
            if not isinstance(tdf_cutoff, (int, float)) or tdf_cutoff < 0:
                self.logger.warning(f'Ignoring illegal value of tdf_cutoff {tdf_cutoff}')
            else:
                tdf[tdf > tdf_cutoff] = np.nan
                self.logger.debug(f'tdf_cutoff = {tdf_cutoff}')

        # Remove nans
        tdf_mean = np.nanmean(tdf)
        self.logger.debug(f'tdf_mean = {tdf_mean}')
        np.nan_to_num(tdf, copy=False, nan=tdf_mean, posinf=tdf_mean, neginf=0.)

        # Plot dark field
        if self.save_figs:
            if self.galaxy_flag:
                quick_imshow(tdf, title='dark field', path='tomo_reduce_plots', save_fig=True,
                        save_only=True)
            else:
                quick_imshow(tdf, title='dark field', path=self.output_folder, save_fig=True,
                        save_only=True)

        # Add dark field to reduced data NXprocess
        reduced_data.data = NXdata()
        reduced_data.data['dark_field'] = tdf

        return(reduced_data)

    def _gen_bright(self, nxentry, reduced_data):
        """Generate bright field.
        """
        from nexusformat.nexus import NXdata

        from CHAP.common.models.map import get_scanparser, import_scanparser

        # Get the bright field images
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key and 'data' in nxentry.instrument.detector:
            field_indices = [index for index, key in enumerate(image_key) if key == 1]
            tbf_stack = nxentry.instrument.detector.data[field_indices,:,:]
            # RV the default NXtomo form does not accomodate bright or dark field stacks
        else:
            import_scanparser(nxentry.instrument.source.attrs['station'],
                    nxentry.instrument.source.attrs['experiment_type'])
            bright_field_scans = nxentry.spec_scans.bright_field
            detector_prefix = str(nxentry.instrument.detector.local_name)
            tbf_stack = []
            for nxsubentry_name, nxsubentry in bright_field_scans.items():
                scan_number = int(nxsubentry_name.split('_')[-1])
                scanparser = get_scanparser(bright_field_scans.attrs['spec_file'], scan_number)
                image_offset = int(nxsubentry.instrument.detector.frame_start_number)
                num_image = len(nxsubentry.sample.rotation_angle)
                tbf_stack.append(scanparser.get_detector_data(detector_prefix,
                        (image_offset, image_offset+num_image)))
            if isinstance(tbf_stack, list):
                assert(len(tbf_stack) == 1) # TODO
                tbf_stack = tbf_stack[0]

        # Take median if more than one image
        """Median or mean: It may be best to try the median because of some image 
           artifacts that arise due to crinkles in the upstream kapton tape windows 
           causing some phase contrast images to appear on the detector.
           One thing that also may be useful in a future implementation is to do a 
           brightfield adjustment on EACH frame of the tomo based on a ROI in the 
           corner of the frame where there is no sample but there is the direct X-ray 
           beam because there is frame to frame fluctuations from the incoming beam. 
           We don’t typically account for them but potentially could.
        """
        from nexusformat.nexus import NXdata

        if tbf_stack.ndim == 2:
            tbf = tbf_stack
        elif tbf_stack.ndim == 3:
            tbf = np.median(tbf_stack, axis=0)
            del tbf_stack
        else:
           raise ValueError(f'Invalid tbf_stack shape ({tbf_stacks.shape})')

        # Subtract dark field
        if 'data' in reduced_data and 'dark_field' in reduced_data.data:
            tbf -= reduced_data.data.dark_field
        else:
            self.logger.warning('Dark field unavailable')

        # Set any non-positive values to one
        # (avoid negative bright field values for spikes in dark field)
        tbf[tbf < 1] = 1

        # Plot bright field
        if self.save_figs:
            if self.galaxy_flag:
                quick_imshow(tbf, title='bright field', path='tomo_reduce_plots', save_fig=True,
                        save_only=True)
            else:
                quick_imshow(tbf, title='bright field', path=self.output_folder, save_fig=True,
                        save_only=True)

        # Add bright field to reduced data NXprocess
        if 'data' not in reduced_data: 
            reduced_data.data = NXdata()
        reduced_data.data['bright_field'] = tbf

        return(reduced_data)

    def _set_detector_bounds(self, nxentry, reduced_data, img_x_bounds=None):
        """Set vertical detector bounds for each image stack.
        Right now the range is the same for each set in the image stack.
        """
        from CHAP.common.models.map import get_scanparser, import_scanparser
        from CHAP.common.utils.general import is_index_range

        if self.test_mode:
            return(tuple(self.test_config['img_x_bounds']))

        # Get the first tomography image and the reference heights
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key and 'data' in nxentry.instrument.detector:
            field_indices = [index for index, key in enumerate(image_key) if key == 0]
            first_image = np.asarray(nxentry.instrument.detector.data[field_indices[0],:,:])
            theta = float(nxentry.sample.rotation_angle[field_indices[0]])
            z_translation_all = nxentry.sample.z_translation[field_indices]
            vertical_shifts = sorted(list(set(z_translation_all)))
            num_tomo_stacks = len(vertical_shifts)
        else:
            import_scanparser(nxentry.instrument.source.attrs['station'],
                    nxentry.instrument.source.attrs['experiment_type'])
            tomo_field_scans = nxentry.spec_scans.tomo_fields
            num_tomo_stacks = len(tomo_field_scans.keys())
            center_stack_index = int(num_tomo_stacks/2)
            detector_prefix = str(nxentry.instrument.detector.local_name)
            vertical_shifts = []
            for i, nxsubentry in enumerate(tomo_field_scans.items()):
                scan_number = int(nxsubentry[0].split('_')[-1])
                scanparser = get_scanparser(tomo_field_scans.attrs['spec_file'], scan_number)
                image_offset = int(nxsubentry[1].instrument.detector.frame_start_number)
                vertical_shifts.append(nxsubentry[1].sample.z_translation)
                if i == center_stack_index:
                    first_image = scanparser.get_detector_data(detector_prefix, image_offset)
                    theta = float(nxsubentry[1].sample.rotation_angle[0])

        # Select image bounds
        title = f'tomography image at theta={round(theta, 2)+0}'
        if img_x_bounds is not None:
            if not is_index_range(img_x_bounds, ge=0, le=first_image.shape[0]):
                raise ValueError(f'Invalid parameter img_x_bounds ({img_x_bounds})')
            #RV TODO make interactive upon request?
            return(img_x_bounds)
        if nxentry.instrument.source.attrs['station'] in ('id1a3', 'id3a'):
            pixel_size = nxentry.instrument.detector.x_pixel_size
            # Try to get a fit from the bright field
            tbf = np.asarray(reduced_data.data.bright_field)
            tbf_shape = tbf.shape
            x_sum = np.sum(tbf, 1)
            x_sum_min = x_sum.min()
            x_sum_max = x_sum.max()
            fit = Fit.fit_data(x_sum, 'rectangle', x=np.array(range(len(x_sum))), form='atan',
                    guess=True)
            parameters = fit.best_values
            x_low_fit = parameters.get('center1', None)
            x_upp_fit = parameters.get('center2', None)
            sig_low = parameters.get('sigma1', None)
            sig_upp = parameters.get('sigma2', None)
            have_fit = fit.success and x_low_fit is not None and x_upp_fit is not None and \
                    sig_low is not None and sig_upp is not None and \
                    0 <= x_low_fit < x_upp_fit <= x_sum.size and \
                    (sig_low+sig_upp)/(x_upp_fit-x_low_fit) < 0.1
            if have_fit:
                # Set a 5% margin on each side
                margin = 0.05*(x_upp_fit-x_low_fit)
                x_low_fit = max(0, x_low_fit-margin)
                x_upp_fit = min(tbf_shape[0], x_upp_fit+margin)
            if num_tomo_stacks == 1:
                if have_fit:
                    # Set the default range to enclose the full fitted window
                    x_low = int(x_low_fit)
                    x_upp = int(x_upp_fit)
                else:
                    # Center a default range of 1 mm (RV: can we get this from the slits?)
                    num_x_min = int((1.0-0.5*pixel_size)/pixel_size)
                    x_low = int(0.5*(tbf_shape[0]-num_x_min))
                    x_upp = x_low+num_x_min
            else:
                # Get the default range from the reference heights
                delta_z = vertical_shifts[1]-vertical_shifts[0]
                for i in range(2, num_tomo_stacks):
                    delta_z = min(delta_z, vertical_shifts[i]-vertical_shifts[i-1])
                self.logger.debug(f'delta_z = {delta_z}')
                num_x_min = int((delta_z-0.5*pixel_size)/pixel_size)
                self.logger.debug(f'num_x_min = {num_x_min}')
                if num_x_min > tbf_shape[0]:
                    self.logger.warning('Image bounds and pixel size prevent seamless stacking')
                if have_fit:
                    # Center the default range relative to the fitted window
                    x_low = int(0.5*(x_low_fit+x_upp_fit-num_x_min))
                    x_upp = x_low+num_x_min
                else:
                    # Center the default range
                    x_low = int(0.5*(tbf_shape[0]-num_x_min))
                    x_upp = x_low+num_x_min
            if self.galaxy_flag:
                img_x_bounds = (x_low, x_upp)
            else:
                tmp = np.copy(tbf)
                tmp_max = tmp.max()
                tmp[x_low,:] = tmp_max
                tmp[x_upp-1,:] = tmp_max
                quick_imshow(tmp, title='bright field')
                tmp = np.copy(first_image)
                tmp_max = tmp.max()
                tmp[x_low,:] = tmp_max
                tmp[x_upp-1,:] = tmp_max
                quick_imshow(tmp, title=title)
                del tmp
                quick_plot((range(x_sum.size), x_sum),
                        ([x_low, x_low], [x_sum_min, x_sum_max], 'r-'),
                        ([x_upp, x_upp], [x_sum_min, x_sum_max], 'r-'),
                        title='sum over theta and y')
                print(f'lower bound = {x_low} (inclusive)')
                print(f'upper bound = {x_upp} (exclusive)]')
                accept =  input_yesno('Accept these bounds (y/n)?', 'y')
                clear_imshow('bright field')
                clear_imshow(title)
                clear_plot('sum over theta and y')
                if accept:
                    img_x_bounds = (x_low, x_upp)
                else:
                    while True:
                        mask, img_x_bounds = draw_mask_1d(x_sum, title='select x data range',
                                legend='sum over theta and y')
                        if len(img_x_bounds) == 1:
                            break
                        else:
                            print(f'Choose a single connected data range')
                    img_x_bounds = tuple(img_x_bounds[0])
            if (num_tomo_stacks > 1 and img_x_bounds[1]-img_x_bounds[0]+1 < 
                    int((delta_z-0.5*pixel_size)/pixel_size)):
                self.logger.warning('Image bounds and pixel size prevent seamless stacking')
        else:
            if num_tomo_stacks > 1:
                raise NotImplementedError('Selecting image bounds for multiple stacks on FMB')
            # For FMB: use the first tomography image to select range
            # RV: revisit if they do tomography with multiple stacks
            x_sum = np.sum(first_image, 1)
            x_sum_min = x_sum.min()
            x_sum_max = x_sum.max()
            if self.galaxy_flag:
                if img_x_bounds is None:
                    img_x_bounds = (0, first_image.shape[0])
            else:
                print('Select vertical data reduction range from first tomography image')
                img_x_bounds = select_image_bounds(first_image, 0, title=title)
                if img_x_bounds is None:
                    raise ValueError('Unable to select image bounds')

        # Plot results
        if self.save_figs:
            if self.galaxy_flag:
                path = 'tomo_reduce_plots'
            else:
                path = self.output_folder
            x_low = img_x_bounds[0]
            x_upp = img_x_bounds[1]
            tmp = np.copy(first_image)
            tmp_max = tmp.max()
            tmp[x_low,:] = tmp_max
            tmp[x_upp-1,:] = tmp_max
            quick_imshow(tmp, title=title, path=path, save_fig=True, save_only=True)
            quick_plot((range(x_sum.size), x_sum),
                    ([x_low, x_low], [x_sum_min, x_sum_max], 'r-'),
                    ([x_upp, x_upp], [x_sum_min, x_sum_max], 'r-'),
                    title='sum over theta and y', path=path, save_fig=True, save_only=True)
            del tmp

        return(img_x_bounds)

    def _set_zoom_or_skip(self):
        """Set zoom and/or theta skip to reduce memory the requirement for the analysis.
        """
#        if input_yesno('\nDo you want to zoom in to reduce memory requirement (y/n)?', 'n'):
#            zoom_perc = input_int('    Enter zoom percentage', ge=1, le=100)
#        else:
#            zoom_perc = None
        zoom_perc = None
#        if input_yesno('Do you want to skip thetas to reduce memory requirement (y/n)?', 'n'):
#            num_theta_skip = input_int('    Enter the number skip theta interval', ge=0,
#                    lt=num_theta)
#        else:
#            num_theta_skip = None
        num_theta_skip = None
        self.logger.debug(f'zoom_perc = {zoom_perc}')
        self.logger.debug(f'num_theta_skip = {num_theta_skip}')

        return(zoom_perc, num_theta_skip)

    def _gen_tomo(self, nxentry, reduced_data):
        """Generate tomography fields.
        """
        import numexpr as ne
        import scipy.ndimage as spi

        from CHAP.common.models.map import get_scanparser, import_scanparser

        # Get full bright field
        tbf = np.asarray(reduced_data.data.bright_field)
        tbf_shape = tbf.shape

        # Get image bounds
        img_x_bounds = tuple(reduced_data.get('img_x_bounds', (0, tbf_shape[0])))
        img_y_bounds = tuple(reduced_data.get('img_y_bounds', (0, tbf_shape[1])))

        # Get resized dark field
#        if 'dark_field' in data:
#            tbf = np.asarray(reduced_data.data.dark_field[
#                    img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]])
#        else:
#            self.logger.warning('Dark field unavailable')
#            tdf = None
        tdf = None

        # Resize bright field
        if img_x_bounds != (0, tbf.shape[0]) or img_y_bounds != (0, tbf.shape[1]):
            tbf = tbf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]

        # Get the tomography images
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key and 'data' in nxentry.instrument.detector:
            field_indices_all = [index for index, key in enumerate(image_key) if key == 0]
            z_translation_all = nxentry.sample.z_translation[field_indices_all]
            z_translation_levels = sorted(list(set(z_translation_all)))
            num_tomo_stacks = len(z_translation_levels)
            tomo_stacks = num_tomo_stacks*[np.array([])]
            horizontal_shifts = []
            vertical_shifts = []
            thetas = None
            tomo_stacks = []
            for i, z_translation in enumerate(z_translation_levels):
                field_indices = [field_indices_all[index]
                        for index, z in enumerate(z_translation_all) if z == z_translation]
                horizontal_shift = list(set(nxentry.sample.x_translation[field_indices]))
                assert(len(horizontal_shift) == 1)
                horizontal_shifts += horizontal_shift
                vertical_shift = list(set(nxentry.sample.z_translation[field_indices]))
                assert(len(vertical_shift) == 1)
                vertical_shifts += vertical_shift
                sequence_numbers = nxentry.instrument.detector.sequence_number[field_indices]
                if thetas is None:
                    thetas = np.asarray(nxentry.sample.rotation_angle[field_indices]) \
                             [sequence_numbers]
                else:
                    assert(all(thetas[i] == nxentry.sample.rotation_angle[field_indices[index]]
                            for i, index in enumerate(sequence_numbers)))
                assert(list(set(sequence_numbers)) == [i for i in range(len(sequence_numbers))])
                if list(sequence_numbers) == [i for i in range(len(sequence_numbers))]:
                    tomo_stack = np.asarray(nxentry.instrument.detector.data[field_indices])
                else:
                    raise ValueError('Unable to load the tomography images')
                tomo_stacks.append(tomo_stack)
        else:
            import_scanparser(nxentry.instrument.source.attrs['station'],
                    nxentry.instrument.source.attrs['experiment_type'])
            tomo_field_scans = nxentry.spec_scans.tomo_fields
            num_tomo_stacks = len(tomo_field_scans.keys())
            center_stack_index = int(num_tomo_stacks/2)
            detector_prefix = str(nxentry.instrument.detector.local_name)
            thetas = None
            tomo_stacks = []
            horizontal_shifts = []
            vertical_shifts = []
            for nxsubentry_name, nxsubentry in tomo_field_scans.items():
                scan_number = int(nxsubentry_name.split('_')[-1])
                scanparser = get_scanparser(tomo_field_scans.attrs['spec_file'], scan_number)
                image_offset = int(nxsubentry.instrument.detector.frame_start_number)
                if thetas is None:
                    thetas = np.asarray(nxsubentry.sample.rotation_angle)
                num_image = len(thetas)
                tomo_stacks.append(scanparser.get_detector_data(detector_prefix,
                        (image_offset, image_offset+num_image)))
                horizontal_shifts.append(nxsubentry.sample.x_translation)
                vertical_shifts.append(nxsubentry.sample.z_translation)

        reduced_tomo_stacks = []
        if self.galaxy_flag:
            path = 'tomo_reduce_plots'
        else:
            path = self.output_folder
        for i, tomo_stack in enumerate(tomo_stacks):
            # Resize the tomography images
            # Right now the range is the same for each set in the image stack.
            if img_x_bounds != (0, tbf.shape[0]) or img_y_bounds != (0, tbf.shape[1]):
                t0 = time()
                tomo_stack = tomo_stack[:,img_x_bounds[0]:img_x_bounds[1],
                        img_y_bounds[0]:img_y_bounds[1]].astype('float64')
                self.logger.debug(f'Resizing tomography images took {time()-t0:.2f} seconds')

            # Subtract dark field
            if tdf is not None:
                t0 = time()
                with set_numexpr_threads(self.num_core):
                    ne.evaluate('tomo_stack-tdf', out=tomo_stack)
                self.logger.debug(f'Subtracting dark field took {time()-t0:.2f} seconds')

            # Normalize
            t0 = time()
            with set_numexpr_threads(self.num_core):
                ne.evaluate('tomo_stack/tbf', out=tomo_stack, truediv=True)
            self.logger.debug(f'Normalizing took {time()-t0:.2f} seconds')

            # Remove non-positive values and linearize data
            t0 = time()
            cutoff = 1.e-6
            with set_numexpr_threads(self.num_core):
                ne.evaluate('where(tomo_stack<cutoff, cutoff, tomo_stack)', out=tomo_stack)
            with set_numexpr_threads(self.num_core):
                ne.evaluate('-log(tomo_stack)', out=tomo_stack)
            self.logger.debug('Removing non-positive values and linearizing data took '+
                    f'{time()-t0:.2f} seconds')

            # Get rid of nans/infs that may be introduced by normalization
            t0 = time()
            np.where(np.isfinite(tomo_stack), tomo_stack, 0.)
            self.logger.debug(f'Remove nans/infs took {time()-t0:.2f} seconds')

            # Downsize tomography stack to smaller size
            # TODO use theta_skip as well
            tomo_stack = tomo_stack.astype('float32')
            if not self.test_mode:
                if len(tomo_stacks) == 1:
                    title = f'red fullres theta {round(thetas[0], 2)+0}'
                else:
                    title = f'red stack {i+1} fullres theta {round(thetas[0], 2)+0}'
                quick_imshow(tomo_stack[0,:,:], title=title, path=path, save_fig=self.save_figs,
                        save_only=self.save_only, block=self.block)
#                if not self.block:
#                    clear_imshow(title)
            if False and zoom_perc != 100:
                t0 = time()
                self.logger.debug(f'Zooming in ...')
                tomo_zoom_list = []
                for j in range(tomo_stack.shape[0]):
                    tomo_zoom = spi.zoom(tomo_stack[j,:,:], 0.01*zoom_perc)
                    tomo_zoom_list.append(tomo_zoom)
                tomo_stack = np.stack([tomo_zoom for tomo_zoom in tomo_zoom_list])
                self.logger.debug(f'... done in {time()-t0:.2f} seconds')
                self.logger.info(f'Zooming in took {time()-t0:.2f} seconds')
                del tomo_zoom_list
                if not self.test_mode:
                    title = f'red stack {zoom_perc}p theta {round(thetas[0], 2)+0}'
                    quick_imshow(tomo_stack[0,:,:], title=title, path=path, save_fig=self.save_figs,
                            save_only=self.save_only, block=self.block)
#                    if not self.block:
#                        clear_imshow(title)

            # Save test data to file
            if self.test_mode:
#                row_index = int(tomo_stack.shape[0]/2)
#                np.savetxt(f'{self.output_folder}/red_stack_{i+1}.txt', tomo_stack[row_index,:,:],
#                        fmt='%.6e')
                row_index = int(tomo_stack.shape[1]/2)
                np.savetxt(f'{self.output_folder}/red_stack_{i+1}.txt', tomo_stack[:,row_index,:],
                        fmt='%.6e')

            # Combine resized stacks
            reduced_tomo_stacks.append(tomo_stack)

        # Add tomo field info to reduced data NXprocess
        reduced_data['rotation_angle'] = thetas
        reduced_data['x_translation'] = np.asarray(horizontal_shifts)
        reduced_data['z_translation'] = np.asarray(vertical_shifts)
        reduced_data.data['tomo_fields'] = np.asarray(reduced_tomo_stacks)

        if tdf is not None:
            del tdf
        del tbf

        return(reduced_data)

    def _find_center_one_plane(self, sinogram, row, thetas, eff_pixel_size, cross_sectional_dim,
            path=None, tol=0.1, num_core=1):
        """Find center for a single tomography plane.
        """
        import tomopy

        # Try automatic center finding routines for initial value
        # sinogram index order: theta,column
        # need column,theta for iradon, so take transpose
        sinogram = np.asarray(sinogram)
        sinogram_T = sinogram.T
        center = sinogram.shape[1]/2

        # Try using Nghia Vo’s method
        t0 = time()
        if num_core > num_core_tomopy_limit:
            self.logger.debug(f'Running find_center_vo on {num_core_tomopy_limit} cores ...')
            tomo_center = tomopy.find_center_vo(sinogram, ncore=num_core_tomopy_limit)
        else:
            self.logger.debug(f'Running find_center_vo on {num_core} cores ...')
            tomo_center = tomopy.find_center_vo(sinogram, ncore=num_core)
        self.logger.debug(f'... done in {time()-t0:.2f} seconds')
        self.logger.info(f'Finding the center using Nghia Vo’s method took {time()-t0:.2f} seconds')
        center_offset_vo = tomo_center-center
        self.logger.info(f'Center at row {row} using Nghia Vo’s method = {center_offset_vo:.2f}')
        t0 = time()
        self.logger.debug(f'Running _reconstruct_one_plane on {self.num_core} cores ...')
        recon_plane = self._reconstruct_one_plane(sinogram_T, tomo_center, thetas,
                eff_pixel_size, cross_sectional_dim, False, num_core)
        self.logger.debug(f'... done in {time()-t0:.2f} seconds')
        self.logger.info(f'Reconstructing row {row} took {time()-t0:.2f} seconds')

        title = f'edges row{row} center offset{center_offset_vo:.2f} Vo'
        self._plot_edges_one_plane(recon_plane, title, path=path)

        # Try using phase correlation method
#        if input_yesno('Try finding center using phase correlation (y/n)?', 'n'):
#            t0 = time()
#            self.logger.debug(f'Running find_center_pc ...')
#            tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=0.1, rotc_guess=tomo_center)
#            error = 1.
#            while error > tol:
#                prev = tomo_center
#                tomo_center = tomopy.find_center_pc(sinogram, sinogram, tol=tol,
#                        rotc_guess=tomo_center)
#                error = np.abs(tomo_center-prev)
#            self.logger.debug(f'... done in {time()-t0:.2f} seconds')
#            self.logger.info('Finding the center using the phase correlation method took '+
#                    f'{time()-t0:.2f} seconds')
#            center_offset = tomo_center-center
#            print(f'Center at row {row} using phase correlation = {center_offset:.2f}')
#            t0 = time()
#            self.logger.debug(f'Running _reconstruct_one_plane on {self.num_core} cores ...')
#            recon_plane = self._reconstruct_one_plane(sinogram_T, tomo_center, thetas,
#                    eff_pixel_size, cross_sectional_dim, False, num_core)
#            self.logger.debug(f'... done in {time()-t0:.2f} seconds')
#            self.logger.info(f'Reconstructing row {row} took {time()-t0:.2f} seconds')
#
#            title = f'edges row{row} center_offset{center_offset:.2f} PC'
#            self._plot_edges_one_plane(recon_plane, title, path=path)

        # Select center location
#        if input_yesno('Accept a center location (y) or continue search (n)?', 'y'):
        if True:
#            center_offset = input_num('    Enter chosen center offset', ge=-center, le=center,
#                    default=center_offset_vo)
            center_offset = center_offset_vo
            del sinogram_T
            del recon_plane
            return float(center_offset)

        # perform center finding search
        while True:
            center_offset_low = input_int('\nEnter lower bound for center offset', ge=-center,
                    le=center)
            center_offset_upp = input_int('Enter upper bound for center offset',
                    ge=center_offset_low, le=center)
            if center_offset_upp == center_offset_low:
                center_offset_step = 1
            else:
                center_offset_step = input_int('Enter step size for center offset search', ge=1,
                        le=center_offset_upp-center_offset_low)
            num_center_offset = 1+int((center_offset_upp-center_offset_low)/center_offset_step)
            center_offsets = np.linspace(center_offset_low, center_offset_upp, num_center_offset)
            for center_offset in center_offsets:
                if center_offset == center_offset_vo:
                    continue
                t0 = time()
                self.logger.debug(f'Running _reconstruct_one_plane on {num_core} cores ...')
                recon_plane = self._reconstruct_one_plane(sinogram_T, center_offset+center, thetas,
                        eff_pixel_size, cross_sectional_dim, False, num_core)
                self.logger.debug(f'... done in {time()-t0:.2f} seconds')
                self.logger.info(f'Reconstructing center_offset {center_offset} took '+
                        f'{time()-t0:.2f} seconds')
                title = f'edges row{row} center_offset{center_offset:.2f}'
                self._plot_edges_one_plane(recon_plane, title, path=path)
            if input_int('\nContinue (0) or end the search (1)', ge=0, le=1):
                break

        del sinogram_T
        del recon_plane
        center_offset = input_num('    Enter chosen center offset', ge=-center, le=center)
        return float(center_offset)

    def _reconstruct_one_plane(self, tomo_plane_T, center, thetas, eff_pixel_size,
            cross_sectional_dim, plot_sinogram=True, num_core=1):
        """Invert the sinogram for a single tomography plane.
        """
        import scipy.ndimage as spi
        from skimage.transform import iradon
        import tomopy

        # tomo_plane_T index order: column,theta
        assert(0 <= center < tomo_plane_T.shape[0])
        center_offset = center-tomo_plane_T.shape[0]/2
        two_offset = 2*int(np.round(center_offset))
        two_offset_abs = np.abs(two_offset)
        max_rad = int(0.55*(cross_sectional_dim/eff_pixel_size)) # 10% slack to avoid edge effects
        if max_rad > 0.5*tomo_plane_T.shape[0]:
            max_rad = 0.5*tomo_plane_T.shape[0]
        dist_from_edge = max(1, int(np.floor((tomo_plane_T.shape[0]-two_offset_abs)/2.)-max_rad))
        if two_offset >= 0:
            self.logger.debug(f'sinogram range = [{two_offset+dist_from_edge}, {-dist_from_edge}]')
            sinogram = tomo_plane_T[two_offset+dist_from_edge:-dist_from_edge,:]
        else:
            self.logger.debug(f'sinogram range = [{dist_from_edge}, {two_offset-dist_from_edge}]')
            sinogram = tomo_plane_T[dist_from_edge:two_offset-dist_from_edge,:]
        if not self.galaxy_flag and plot_sinogram:
            quick_imshow(sinogram.T, f'sinogram center offset{center_offset:.2f}', aspect='auto',
                    path=self.output_folder, save_fig=self.save_figs, save_only=self.save_only,
                    block=self.block)

        # Inverting sinogram
        t0 = time()
        recon_sinogram = iradon(sinogram, theta=thetas, circle=True)
        self.logger.debug(f'Inverting sinogram took {time()-t0:.2f} seconds')
        del sinogram

        # Performing Gaussian filtering and removing ring artifacts
        recon_parameters = None#self.config.get('recon_parameters')
        if recon_parameters is None:
            sigma = 1.0
            ring_width = 15
        else:
            sigma = recon_parameters.get('gaussian_sigma', 1.0)
            if not is_num(sigma, ge=0.0):
                self.logger.warning(f'Invalid gaussian_sigma ({sigma}) in _reconstruct_one_plane, '+
                        'set to a default value of 1.0')
                sigma = 1.0
            ring_width = recon_parameters.get('ring_width', 15)
            if not isinstance(ring_width, int) or ring_width < 0:
                self.logger.warning(f'Invalid ring_width ({ring_width}) in '+
                        '_reconstruct_one_plane, set to a default value of 15')
                ring_width = 15
        t0 = time()
        recon_sinogram = spi.gaussian_filter(recon_sinogram, sigma, mode='nearest')
        recon_clean = np.expand_dims(recon_sinogram, axis=0)
        del recon_sinogram
        recon_clean = tomopy.misc.corr.remove_ring(recon_clean, rwidth=ring_width, ncore=num_core)
        self.logger.debug(f'Filtering and removing ring artifacts took {time()-t0:.2f} seconds')

        return recon_clean

    def _plot_edges_one_plane(self, recon_plane, title, path=None):
        from skimage.restoration import denoise_tv_chambolle

        vis_parameters = None#self.config.get('vis_parameters')
        if vis_parameters is None:
            weight = 0.1
        else:
            weight = vis_parameters.get('denoise_weight', 0.1)
            if not is_num(weight, ge=0.0):
                self.logger.warning(f'Invalid weight ({weight}) in _plot_edges_one_plane, '+
                        'set to a default value of 0.1')
                weight = 0.1
        edges = denoise_tv_chambolle(recon_plane, weight=weight)
        vmax = np.max(edges[0,:,:])
        vmin = -vmax
        if path is None:
            path = self.output_folder
        quick_imshow(edges[0,:,:], f'{title} coolwarm', path=path, cmap='coolwarm',
                save_fig=self.save_figs, save_only=self.save_only, block=self.block)
        quick_imshow(edges[0,:,:], f'{title} gray', path=path, cmap='gray', vmin=vmin, vmax=vmax,
                save_fig=self.save_figs, save_only=self.save_only, block=self.block)
        del edges

    def _reconstruct_one_tomo_stack(self, tomo_stack, thetas, center_offsets=[], num_core=1,
            algorithm='gridrec'):
        """Reconstruct a single tomography stack.
        """
        import tomopy

        # tomo_stack order: row,theta,column
        # input thetas must be in degrees 
        # centers_offset: tomography axis shift in pixels relative to column center
        # RV should we remove stripes?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html
        # RV should we remove rings?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.misc.corr.html
        # RV: Add an option to do (extra) secondary iterations later or to do some sort of convergence test?
        if not len(center_offsets):
            centers = np.zeros((tomo_stack.shape[0]))
        elif len(center_offsets) == 2:
            centers = np.linspace(center_offsets[0], center_offsets[1], tomo_stack.shape[0])
        else:
            if center_offsets.size != tomo_stack.shape[0]:
                raise ValueError('center_offsets dimension mismatch in reconstruct_one_tomo_stack')
            centers = center_offsets
        centers += tomo_stack.shape[2]/2

        # Get reconstruction parameters
        recon_parameters = None#self.config.get('recon_parameters')
        if recon_parameters is None:
            sigma = 2.0
            secondary_iters = 0
            ring_width = 15
        else:
            sigma = recon_parameters.get('stripe_fw_sigma', 2.0)
            if not is_num(sigma, ge=0):
                self.logger.warning(f'Invalid stripe_fw_sigma ({sigma}) in '+
                        '_reconstruct_one_tomo_stack, set to a default value of 2.0')
                ring_width = 15
            secondary_iters = recon_parameters.get('secondary_iters', 0)
            if not isinstance(secondary_iters, int) or secondary_iters < 0:
                self.logger.warning(f'Invalid secondary_iters ({secondary_iters}) in '+
                        '_reconstruct_one_tomo_stack, set to a default value of 0 (skip them)')
                ring_width = 0
            ring_width = recon_parameters.get('ring_width', 15)
            if not isinstance(ring_width, int) or ring_width < 0:
                self.logger.warning(f'Invalid ring_width ({ring_width}) in '+
                        '_reconstruct_one_plane, set to a default value of 15')
                ring_width = 15

        # Remove horizontal stripe
        t0 = time()
        if num_core > num_core_tomopy_limit:
            self.logger.debug('Running remove_stripe_fw on {num_core_tomopy_limit} cores ...')
            tomo_stack = tomopy.prep.stripe.remove_stripe_fw(tomo_stack, sigma=sigma,
                    ncore=num_core_tomopy_limit)
        else:
            self.logger.debug(f'Running remove_stripe_fw on {num_core} cores ...')
            tomo_stack = tomopy.prep.stripe.remove_stripe_fw(tomo_stack, sigma=sigma,
                    ncore=num_core)
        self.logger.debug(f'... tomopy.prep.stripe.remove_stripe_fw took {time()-t0:.2f} seconds')

        # Perform initial image reconstruction
        self.logger.debug('Performing initial image reconstruction')
        t0 = time()
        self.logger.debug(f'Running recon on {num_core} cores ...')
        tomo_recon_stack = tomopy.recon(tomo_stack, np.radians(thetas), centers,
                sinogram_order=True, algorithm=algorithm, ncore=num_core)
        self.logger.debug(f'... done in {time()-t0:.2f} seconds')
        self.logger.info(f'Performing initial image reconstruction took {time()-t0:.2f} seconds')

        # Run optional secondary iterations
        if secondary_iters > 0:
            self.logger.debug(f'Running {secondary_iters} secondary iterations')
            #options = {'method':'SIRT_CUDA', 'proj_type':'cuda', 'num_iter':secondary_iters}
            #RV: doesn't work for me:
            #"Error: CUDA error 803: system has unsupported display driver/cuda driver combination."
            #options = {'method':'SIRT', 'proj_type':'linear', 'MinConstraint': 0, 'num_iter':secondary_iters}
            #SIRT did not finish while running overnight
            #options = {'method':'SART', 'proj_type':'linear', 'num_iter':secondary_iters}
            options = {'method':'SART', 'proj_type':'linear', 'MinConstraint': 0,
                    'num_iter':secondary_iters}
            t0 = time()
            self.logger.debug(f'Running recon on {num_core} cores ...')
            tomo_recon_stack  = tomopy.recon(tomo_stack, np.radians(thetas), centers,
                    init_recon=tomo_recon_stack, options=options, sinogram_order=True,
                    algorithm=tomopy.astra, ncore=num_core)
            self.logger.debug(f'... done in {time()-t0:.2f} seconds')
            self.logger.info(f'Performing secondary iterations took {time()-t0:.2f} seconds')

        # Remove ring artifacts
        t0 = time()
        tomopy.misc.corr.remove_ring(tomo_recon_stack, rwidth=ring_width, out=tomo_recon_stack,
                ncore=num_core)
        self.logger.debug(f'Removing ring artifacts took {time()-t0:.2f} seconds')

        return tomo_recon_stack

    def _resize_reconstructed_data(self, data, x_bounds=None, y_bounds=None, z_bounds=None,
            z_only=False):
        """Resize the reconstructed tomography data.
        """
        # Data order: row(z),x,y or stack,row(z),x,y
        if isinstance(data, list):
            for stack in data:
                assert(stack.ndim == 3)
            num_tomo_stacks = len(data)
            tomo_recon_stacks = data
        else:
            assert(data.ndim == 3)
            num_tomo_stacks = 1
            tomo_recon_stacks = [data]

        if x_bounds == (-1, -1):
            x_bounds = None
        elif not z_only and x_bounds is None:
            # Selecting x bounds (in yz-plane)
            tomosum = 0
            [tomosum := tomosum+np.sum(tomo_recon_stacks[i], axis=(0,2))
                    for i in range(num_tomo_stacks)]
            select_x_bounds = input_yesno('\nDo you want to change the image x-bounds (y/n)?', 'y')
            if not select_x_bounds:
                x_bounds = None
            else:
                accept = False
                index_ranges = None
                while not accept:
                    mask, x_bounds = draw_mask_1d(tomosum, current_index_ranges=index_ranges,
                            title='select x data range', legend='recon stack sum yz')
                    while len(x_bounds) != 1:
                        print('Please select exactly one continuous range')
                        mask, x_bounds = draw_mask_1d(tomosum, title='select x data range',
                                legend='recon stack sum yz')
                    x_bounds = x_bounds[0]
                    accept = True
            self.logger.debug(f'x_bounds = {x_bounds}')

        if y_bounds == (-1, -1):
            y_bounds = None
        elif not z_only and y_bounds is None:
            # Selecting y bounds (in xz-plane)
            tomosum = 0
            [tomosum := tomosum+np.sum(tomo_recon_stacks[i], axis=(0,1))
                    for i in range(num_tomo_stacks)]
            select_y_bounds = input_yesno('\nDo you want to change the image y-bounds (y/n)?', 'y')
            if not select_y_bounds:
                y_bounds = None
            else:
                accept = False
                index_ranges = None
                while not accept:
                    mask, y_bounds = draw_mask_1d(tomosum, current_index_ranges=index_ranges,
                            title='select x data range', legend='recon stack sum xz')
                    while len(y_bounds) != 1:
                        print('Please select exactly one continuous range')
                        mask, y_bounds = draw_mask_1d(tomosum, title='select x data range',
                                legend='recon stack sum xz')
                    y_bounds = y_bounds[0]
                    accept = True
            self.logger.debug(f'y_bounds = {y_bounds}')

        # Selecting z bounds (in xy-plane) (only valid for a single image stack)
        if z_bounds == (-1, -1):
            z_bounds = None
        elif z_bounds is None and num_tomo_stacks != 1:
            tomosum = 0
            [tomosum := tomosum+np.sum(tomo_recon_stacks[i], axis=(1,2))
                    for i in range(num_tomo_stacks)]
            select_z_bounds = input_yesno('Do you want to change the image z-bounds (y/n)?', 'n')
            if not select_z_bounds:
                z_bounds = None
            else:
                accept = False
                index_ranges = None
                while not accept:
                    mask, z_bounds = draw_mask_1d(tomosum, current_index_ranges=index_ranges,
                            title='select x data range', legend='recon stack sum xy')
                    while len(z_bounds) != 1:
                        print('Please select exactly one continuous range')
                        mask, z_bounds = draw_mask_1d(tomosum, title='select x data range',
                                legend='recon stack sum xy')
                    z_bounds = z_bounds[0]
                    accept = True
            self.logger.debug(f'z_bounds = {z_bounds}')

        return(x_bounds, y_bounds, z_bounds)


if __name__ == '__main__':
    from CHAP.processor import main
    main()
