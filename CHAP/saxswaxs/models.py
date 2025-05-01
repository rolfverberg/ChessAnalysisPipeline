"""Definition and utilities for integration tools"""

# System modules
from copy import deepcopy
import os
from typing import (
    Literal,
    Optional,
)

# Third party modules
import numpy as np
from pydantic import (
    ConfigDict,
    Field,
    FilePath,
    PrivateAttr,
    StringConstraints,
    conlist,
    field_validator,
    model_validator,
    validator,
)
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from typing_extensions import Annotated

# Local modules
from CHAP.giwaxs.models import (
    PyfaiIntegratorConfig,
    PyfaiIntegrationConfig,
)


class PyfaiIntegratorZarrConfig(PyfaiIntegratorConfig):

    _placeholder_result: PrivateAttr = None

#    _use_CDC: PrivateAttr = False #RV
#
#    def integrate(self, ais, input_data):
#        if self._use_CDC:
#            # Adjust azimuthal angles used (since pyfai has some odd conventions)
#            chi_min, chi_max = self.multi_geometry.get('azimuth_range',
#                                                       (-180.0, 180.0))
#            # Force a right-handed coordinate system 
#            chi_min, chi_max = 360 - chi_max, 360 - chi_min
#    
#            # If the discontinuity is crossed, artificially rotate the
#            # detectors to achieve a continuous azimuthal integration range 
#            chi_disc = self.multi_geometry.get('chi_disc', 180)
#            if chi_min < chi_disc and chi_max > chi_disc:
#                chi_offset = chi_max - chi_disc
#            else:
#                chi_offset = 0
#            chi_min -= chi_offset
#            chi_max -= chi_offset
#            for ai in ais.values():
#                ai.rot3 += chi_offset * np.pi/180.0
#            # Use adjusted azimuthal integration range
#            self.multi_geometry['azimuth_range'] = (chi_min, chi_max)
#
##        dummy = input_data['PIL9']
##        print(f'\n\ninput_data {type(dummy)}: {dummy.shape} {dummy.sum()}')
##        for d in dummy:
##            print(f'\t{d.shape} {d.sum()}')
##        print('\n\n')
#        npts = [len(input_data[name])
#                for name in self.integration_params['lst_data']]
#        if not all([_npts == npts[0] for _npts in npts]):
#            raise RuntimeError(
#                'Different number of frames of detector data provided')
#        npts = npts[0]
#        integration_params = {k: v
#                              for k, v in self.integration_params.items()
#                              if k != 'lst_data'}
#        if self.integration_method == 'integrate_radial':
#            from pyFAI.containers import Integrate1dResult
#            ais, _, chi_offset = self.adjust_azimuthal_integrators(ais)
#            results = None
#            for name in self.integration_params.get('lst_data', []):
#                ai = ais[name]
#                _results = [
#                    ai.integrate_radial(
#                        data=input_data[name][i], **integration_params)
#                    for i in range(npts)
#                ]
#                if results is None:
#                    results = _results
#                else:
#                    results = [
#                        Integrate1dResult(
#                            radial=_results[i].radial,
#                            intensity=(_results[i].intensity
#                                       + results[i].intensity))
#                        for i in range(npts)
#                    ]
#            if self._use_CDC:
#                if self.right_handed:
#                    results = [
#                        Integrate1dResult(
#                            radial=r.radial,
#                            intensity=np.flip(r.intensity)
#                        )
#                        for r in results
#                    ]
#                results = [
#                    Integrate1dResult(
#                        radial=r.radial + chi_offset,
#                        intensity=np.where(r.intensity==0, np.nan, r.intensity)
#                    )
#                    for r in results
#                ]
#            return results
#        else:
#            from pyFAI.multi_geometry import MultiGeometry
#            ais = [ais[ai] for ai in self.multi_geometry['ais']]
#            multi_geometry = deepcopy(self.multi_geometry)
#            del multi_geometry['ais']
#            mg = MultiGeometry(ais, **multi_geometry)
#            integration_method = getattr(mg, self.integration_method)
#            integration_params = {k: v
#                                  for k, v in self.integration_params.items()
#                                  if k != 'lst_data'}
#            npts = [len(input_data[name])
#                        for name in self.integration_params['lst_data']]
#            if not all([_npts == npts[0] for _npts in npts]):
#                raise RuntimeError(
#                    'Different number of frames of detector data provided')
#            npts = npts[0]
#            lst_data = [[np.nan_to_num(input_data[name][i]).astype(np.float64)
#                         for name in self.integration_params['lst_data']]
#                        for i in range(npts)]
##            print(f'lst_data {type(lst_data)}: {len(lst_data)}')
##            for d in lst_data:
##                for dd in d:
##                    print(f'\t{dd.shape} {dd.sum()}')
#            results = [
#                integration_method(lst_data=lst_data[i], **integration_params)
#                for i in range(npts)
#            ]
#            if self._use_CDC:
#                if self.integration_method == 'integrate2d' and self.right_handed:
#                    # Flip results along azimuthal axis
#                    from pyFAI.containers import Integrate2dResult
#                    results = [
#                        Integrate2dResult(
#                            np.flip(result.intensity, axis=0),
#                            result.radial, result.azimuthal
#                        )
#                        for result in results
#                    ]
##            dummy = [v.intensity for v in results]
##            print(f'\n\nintensities {type(dummy)}: {len(dummy)}')
##            for d in dummy:
##                print(f'\t{d.shape} {d.sum()}')
##            print('\n\n')
#            return results
#            # Integrate first frame, extract objects to perform
#            # integrations quicker from the result.
#            results = [None] * npts
#            results[0] = integration_method(
#                lst_data=[input_data[name][0]
#                          for name in self.integration_params['lst_data']],
#                **integration_params)
#            engine = mg.engines[results[0].method].engine
#            omega = mg.solidAngleArray()
#            for i in range(1, npts):
#                results[i] = engine.integrate_ng(
#                    lst_data=[input_data[name][i]
#                              for name in self.integration_params['lst_data']],
#                    **integration+params, solidangle=omega
#                )
#            return results
#
#    def init_placeholder_results(self, ais):
#
#        placeholder_result = None
#        placeholder_data = [
#            np.full(ais[name].detector.shape, 0)
#            for name in self.multi_geometry.ais]
#        if self.integration_method == 'integrate_radial':
#            # Radial integration happens detector-by-detector with the
#            # same aprameters for integration range and number of
#            # points. So, for getting radially integrated placeholder
#            # results, only need to perform integration for at most
#            # one detector since it's all 0s anyways.
#            raise RuntimeError('integrate_radial not updated yet')
#            det_name = self.integration_params.get('lst_data', [])[0]
#            ai = ais[det_name]
#            integration_params = {k: v
#                                  for k, v in self.integration_params.items()
#                                  if k != 'lst_data'}
#            placeholder_result = ai.integrate_radial(
#                data=placeholder_data[0], **integration_params)
#        else:
#            mg = self.get_multi_geometry(ais)
#            integration_method = getattr(mg, self.integration_method)
#            integration_params = {k: v
#                                  for k, v in self.integration_params.items()
#                                  if k != 'lst_data'}
#            placeholder_result = integration_method(
#                lst_data=placeholder_data, **integration_params)
#            # right handed coorinate system acieved by flipping
#            # intensity results, not the coordinate axes, so no need
#            # to do this with placeholder results since they're all 0s
#            # anyways.
#
#        self._placeholder_result = placeholder_result
#
#    def adjust_azimuthal_integrators(self, azimuthal_integrators):
#        from pyFAI.multi_geometry import MultiGeometry
#
#        # Adjust azimuthal angles used (since pyfai has some odd conventions)
#        chi_min, chi_max = self.multi_geometry.get('azimuth_range',
#                                                   (-180.0, 180.0))
#        if not self._use_CDC:
#            return azimuthal_integrators, (chi_min, chi_max), 0
#        # Force a right-handed coordinate system
#        chi_min, chi_max = 360 - chi_max, 360 - chi_min
#
#        # If the discontinuity is crossed, artificially rotate the
#        # detectors to achieve a continuous azimuthal integration range
#        chi_disc = self.multi_geometry.get('chi_disc', 180)
#        if chi_min < chi_disc and chi_max > chi_disc:
#            chi_offset = chi_max - chi_disc
#        else:
#            chi_offset = 0
#        chi_min -= chi_offset
#        chi_max -= chi_offset
#        for name, ai in azimuthal_integrators.items():
#            ai.rot3 += chi_offset * np.pi/180.0
#        return azimuthal_integrators, (chi_min, chi_max), chi_offset
#
#    def get_multi_geometry(self, azimuthal_integrators):
#        from pyFAI.multi_geometry import MultiGeometry
#
#        azimuthal_integrators, azimuth_range, _ = \
#            self.adjust_azimuthal_integrators(azimuthal_integrators)
#        ais = [azimuthal_integrators[name]
#               for name in self.multi_geometry['ais']]
#        kwargs = {k: v for k, v in self.multi_geometry.items()
#                  if k not in ('ais', 'azimuth_range')}
#        return MultiGeometry(ais, azimuth_range=azimuth_range, **kwargs)
#
#    def result_axes(self):
#        return []
#
    @property
    def result_shape(self):
        if self.integration_method == 'integrate_radial':
            return (self.integration_params.npt, )
        elif self.integration_method == 'integrate1d':
            return (self.integration_params.npt, )
        elif self.integration_method == 'integrate2d':
            return (self.integration_params.npt_azim,
                    self.integration_params.npt_rad)
        else:
            raise NotImplementedError(
                f'Unimplemented integration_method: {self.integration_method}')

    @property
    def result_coords(self):
        # Third party modules
        from nexusformat.nexus import NXfield
        from pyFAI.gui.utils.units import Unit
#        import pyFAI.containers
#        import pyFAI.units

        if self._placeholder_result is None:
            raise RuntimeError('Missing placeholder results')

#        coords = {}
#        if isinstance(self._placeholder_result,
#                        pyFAI.containers.Integrate2dResult):
#            radial_unit = pyFAI.units.to_unit(
#                self._placeholder_result.radial_unit,
#                type_=pyFAI.units.RADIAL_UNITS)
#            coords[radial_unit.name] = {
#                'attributes': {
#                    'units': radial_unit.unit_symbol,
#                    'long_name': radial_unit.label,
#                },
#                'data': self._placeholder_result.radial.tolist(),
#                'shape': self._placeholder_result.radial.shape,
#                'dtype': 'float32',
#            }
#            azimuthal_unit = pyFAI.units.to_unit(
#                self._placeholder_result.azimuthal_unit,
#                type_=pyFAI.units.AZIMUTHAL_UNITS)
#            coords[azimuthal_unit.name] = {
#                'attributes': {
#                    'units': azimuthal_unit.name.split('_')[-1],
#		    'long_name': azimuthal_unit.label,
#                },
#                'data': self._placeholder_result.azimuthal.tolist(),
#                'shape': self._placeholder_result.azimuthal.shape,
#                'dtype': 'float32',
#            }
#        elif isinstance(self._placeholder_result,
#                        pyFAI.containers.Integrate1dResult):
#            # Integrate1dResult's "radial" property is misleadingly
#            # named here. When using integrate_radial, the property
#            # actually contains azimuthal coordinate values.
#            if self.integration_method == 'integrate_radial':
#                azimuthal_unit = pyFAI.units.to_unit(
#                    self._placeholder_result.unit,
#                    type_=pyFAI.units.AZIMUTHAL_UNITS)
#                coords[azimuthal_unit.name] = {
#                    'attributes': {
#                        'units': azimuthal_unit.name.split('_')[-1],
#                        'long_name': azimuthal_unit.label,
#                    },
#                    'data': self._placeholder_result.radial.tolist(),
#                    'shape': self._placeholder_result.radial.shape,
#                    'dtype': 'float32',
#                }
#            elif self.integration_method == 'integrate1d':
#                radial_unit = pyFAI.units.to_unit(
#                    self.multi_geometry.get('unit', '2th_deg'),
#                    type_=pyFAI.units.RADIAL_UNITS)
#                coords[radial_unit.name] = {
#                    'attributes': {
#                        'units': radial_unit.unit_symbol,
#                        'long_name': radial_unit.label,
#                    },
#                    'data': self._placeholder_result.radial.tolist(),
#                    'shape': self._placeholder_result.radial.shape,
#                    'dtype': 'float32',
#                }
#            else:
#                raise ValueError
#        else:
#            raise TypeError

        coords = {}
        results = self._placeholder_result
        if ('azimuthal' in results
                and results['azimuthal']['unit'] == 'chi_deg'):
            chi = results['azimuthal']['coords']
            if integration.right_handed:
                chi = -np.flip(chi)
                intensities = np.flip(intensities, (len(coords)))
            coords['chi'] = NXfield(chi, 'chi', attrs={'units': 'deg'})
        if results['radial']['unit'] == 'q_A^-1':
            unit = Unit.INV_ANGSTROM.symbol
            coords['q'] = NXfield(results['radial']['coords'], 'q',
                                  attrs={'units': unit})
        else:
            coords['r'] = NXfield(results['radial']['coords'], 'r')#,
#                                  attrs={'units': '\u212b'})
            self.logger.warning(
                f'Unknown radial unit: {results["radial"]["unit"]}')

        return coords

    def get_axes_indices(self, dataset_ndims):
        return {k: dataset_ndims + i
                for i, k in enumerate(self.result_coords.keys())}

    def zarr_tree(self, dataset_shape, dataset_chunks):
        # Third party modules
        import json

#        print(f'\n\n---------> self.result_shape: {self.result_shape}')
#        print(f'\n---------> self.result_coords: {self.result_coords}\n\n')
        tree = {
            # NXprocess
            'attributes': {
                'default': 'data',
                # 'config': json.dumps(self.dict())
            },
            'children': {
                'data': {
                    # NXdata
                    'attributes': {
                        # 'axes': self.result_axes(),
                        **self.get_axes_indices(len(dataset_shape))
                    },
                    'children': {
                        'I': {
                            # NXfield
                            'attributes': {
                                'long_name': 'Intensity (a.u)',
                                'units': 'a.u'
                            },
                            'dtype': 'float64',
                            'shape': (*dataset_shape, *self.result_shape),
                            'chunks': (*dataset_chunks, *self.result_shape),
                            'compressors': None,
                        },
                        **self.result_coords,
                    }   
                }
            }
        }
        return tree


class PyfaiIntegrationZarrConfig(PyfaiIntegrationConfig):
    integrations: conlist(min_length=1, item_type=PyfaiIntegratorZarrConfig)

    def zarr_tree(self, dataset_shape, dataset_chunks):
        ais = {ai.id: ai.ai for ai in self.azimuthal_integrators}

        for integration in self.integrations:
            data = {ai:np.full(ais[ai].detector.shape, 0)
                    for ai in integration.multi_geometry.ais}
#            print(f'\n\ndata:\n{data}')
            integration._placeholder_result = integration.integrate(ais, data)
#            print(f'\n\nintegration {type(integration)}:\n{integration}\n\n')
#            print(f'\n\nintegration._placeholder_result {type(integration._placeholder_result)}:\n{integration._placeholder_result}\n\n')
        tree = {
            'root': {
                'attributes': {
                    'description': 'Container for processed SAXS/WAXS data'
                },
                'children': {
                    'entry': {
                        # NXentry
                        'attributes': {},
                        'children': {
                            'data': {
                                # NXdata
                                'attributes': {
                                    'axes': []
                                },
                                'children': {
                                    'spec_file': {
                                        'shape': dataset_shape,
                                        'dtype': 'b',
                                        'chunks': dataset_chunks,
                                    },
                                    'scan_number': {
                                        'shape': dataset_shape,
                                        'dtype': 'uint8',
                                        'chunks': dataset_chunks,
                                    },
                                    'scan_step_index': {
                                        'shape': dataset_shape,
                                        'dtype': 'uint64',
                                        'chunks': dataset_chunks,
                                    },
                                }
                            }
                        }
                    },
                    **{integration.name: integration.zarr_tree(
                        dataset_shape, dataset_chunks)
                       for integration in self.integrations},
                }
            }
        }
        return tree
