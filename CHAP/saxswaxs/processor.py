#!/usr/bin/env python
"""Processors used only by SAXSWAXS experiments."""

import numpy as np

from CHAP import Processor

class PyfaiIntegrationZarrProcessor(Processor):
    """Processor for performing pyFAI integrations."""
    def process(
            self, data, config=None,
            idx_slices=[{'start':0, 'end': -1, 'step': 1}]):
        # Third party modules
        import fabio

        # Load the validated integration configuration
        config = self.get_config(
            #data=data, config=config, inputdir=inputdir,
            data=data, config=config,
            schema='saxswaxs.models.PyfaiIntegrationZarrConfig')

        # Organize inputs for integrations
        data = {d['name']: d['data']
            for d in [
                d for d in data if isinstance(d['data'], np.ndarray)]}
        ais = {ai.id: ai.ai for ai in config.azimuthal_integrators}

        # Read the mask(s)
        masks = {}
        for ai in config.azimuthal_integrators:
            self.logger.debug(f'Reading {ai.mask_file}')
            try:
                with fabio.open(ai.mask_file) as f:
                    mask = f.data
                    self.logger.debug(f'mask shape for {ai.id}: {mask.shape}')
                    masks[ai.id] = mask
            except:
                self.logger.debug(
                    f'Unable to read mask file for {ai.id} ({ai.mask_file})')
        if not masks:
            masks = None

        # Finalize idx slice for results
        idx = tuple(
            slice(idx_slice.get('start'), idx_slice.get('end'),
            idx_slice.get('step')) for idx_slice in idx_slices)

        # Perform integration(s), package results for ZarrResultsWriter
        results = []
        for integration in config.integrations:
            self.logger.info(f'Integrating {integration.name}...')
            result = integration.integrate(ais, data, masks)
#            print(f'\n\nHERE result:\n{result.keys()} {np.asarray(result["intensities"]).shape}\n\n')
            results.extend(
                [
                    {
                        'path': f'{integration.name}/data/I',
                        'idx': idx,
                        'data': np.asarray([result['intensities']]),
                    },
                ]
            )
#RV
#            from nexusformat.nexus import NXdata, NXfield
#            return NXdata(
#                NXfield(np.asarray(result['intensities']), 'I'),
#                NXfield(np.asarray(result['radial']['coords']), 'q'))
        return results


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
