#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test functions for the Tomo workflow."""

# System modules
import os
import shutil

# Third party modules
import pytest

# Local modules
from CHAP.pipeline import (
    PipelineData,
    PipelineItem,
)
from CHAP.common.processor import MapProcessor
from CHAP.common.reader import (
    SpecReader,
    YAMLReader,
)
from CHAP.common.writer import FileTreeWriter
from CHAP.tomo.processor import *


map_config = {
    'title': 'hollow_cube',
    'station': 'id3b',
    'experiment_type': 'TOMO',
    'sample': {'name': 'hollow_cube'},
    'spec_scans': [{
        'spec_file': 'raw/hollow_cube/hollow_cube',
        'scan_numbers': 3}],
    'independent_dimensions': [
        {'label': 'rotation_angles',
          'units': 'degrees',
          'data_type': 'scan_column',
          'name': 'theta'},
        {'label': 'x_translation',
          'units': 'mm',
          'data_type': 'spec_motor',
          'name': 'GI_samx'},
        {'label': 'z_translation',
          'units': 'mm',
          'data_type': 'spec_motor',
          'name': 'GI_samz'}],
}

def load_detector_config():
    return YAMLReader.run(filename='detector_cube.yaml', log_level='WARNING')

class TestEdd:

    def test_id3b(self):
        detector_config = load_detector_config()

        simfield = TomoSimFieldProcessor.run(
            data=[PipelineData(
                name='YAMLReader',
                data=detector_config,
                schema='common.models.map.DetectorConfig')],
            config={
                'station': map_config['station'],
                'sample_type': map_config['title'],
                'sample_size': [1.0],
                'wall_thickness': 0.2,
                'theta_step': 1.0,
                'slit_size': 2.0,
            },
            log_level='WARNING')
        data = [PipelineData(
            name='TomoSimFieldProcessor',
            data=simfield,
            schema='tomo.models.TomoSimField')]

        darkfield = TomoDarkFieldProcessor.run(data=data, log_level='WARNING')
        data.append(PipelineData(
            name='TomoDarkFieldProcessor',
            data=darkfield,
            schema='tomo.models.TomoDarkField'))

        brightfield = TomoBrightFieldProcessor.run(
            data=data, num_image=10, log_level='WARNING')
        data.append(PipelineData(
            name='TomoBrightFieldProcessor',
            data=brightfield,
            schema='tomo.models.TomoBrightField'))

        tomospec = TomoSpecProcessor.run(data=data, log_level='WARNING')
        FileTreeWriter.run(
            data=[PipelineData(data=tomospec)],
            force_overwrite=True,
            outputdir='raw/hollow_cube',
            log_level='WARNING')

        map_hollow_cube = MapProcessor.run(
            config=map_config,
            detector_config=detector_config,
            log_level='WARNING')
        data = [PipelineData(
            name='MapProcessor', data=map_hollow_cube, schema='tomofields')]

        darkfield = SpecReader.run(
            config={
                'station': map_config['station'],
                'experiment_type': map_config['experiment_type'],
                'sample': map_config['sample'],
                'spec_scans': [
                    {'spec_file': map_config['spec_scans'][0].spec_file,
                     'scan_numbers': 1}],
            },
            detector_config=detector_config,
            log_level='WARNING')
        data.append(PipelineData(
            name='SpecReader', data=darkfield, schema='darkfield'))

        brightfield = SpecReader.run(
            config={
                'station': map_config['station'],
                'experiment_type': map_config['experiment_type'],
                'sample': map_config['sample'],
                'spec_scans': [
                    {'spec_file': map_config['spec_scans'][0].spec_file,
                     'scan_numbers': 2}],
            },
            detector_config=detector_config,
            log_level='WARNING')
        data.append(PipelineData(
            name='SpecReader', data=brightfield, schema='brightfield'))

        data = list(TomoCHESSMapConverter.run(data=data, log_level='WARNING'))

        data += list(TomoReduceProcessor.run(
            data=data,
            config={'img_row_bounds': [3, 35]},
            save_figures=False,
            interactive=False,
            log_level='WARNING'))

        data += TomoFindCenterProcessor.run(
            data=data,
            config={
                'center_rows': [11, 28],
                'gaussian_sigma': 0.05,
                'ring_width': 1,
            },
            save_figures=False,
            interactive=False,
            log_level='WARNING')

        data += TomoReconstructProcessor.run(
            data=data,
            config={
                'x_bounds': [15, 390],
                'y_bounds': [25, 380],
                'secondary_iters': 10,
                'ring_width': 1,
            },
            save_figures=False,
            interactive=False,
            log_level='WARNING')

        tomodata = PipelineItem.get_data(data, schema='tomodata')
        nxentry = tomodata[tomodata.default]
        nxdata = nxentry[nxentry.default]
        reconstructed_data = nxdata.nxsignal
        assert reconstructed_data.shape == (32, 355, 375)
        assert pytest.approx(reconstructed_data.sum()) == 164.28904724121094

        metadata = PipelineItem.get_data(
            data, schema='foxden.reader.FoxdenMetadataReader')
        assert metadata == {
            'btr': 'unknown',
            'did': '/workflow=tomo_reconstruct',
            'parent_did': None,
            'schema': 'user',
            'user_metadata': {
                'findcenter': {
                    'center_offset_max': None,
                    'center_offset_min': None,
                    'center_offsets': [-0.5, -0.5],
                    'center_rows': [11, 28],
                    'center_search_range': None,
                    'center_stack_index': 0,
                    'gaussian_sigma': 0.05,
                    'ring_width': 1.0},
                'reconstructed_data': {
                    'gaussian_sigma': None,
                    'remove_stripe_sigma': None,
                    'ring_width': 1.0,
                    'secondary_iters': 10,
                    'x_bounds': [15, 390],
                    'y_bounds': [25, 380],
                    'z_bounds': [0, 32]},
                'reduced_data': {
                    'delta_theta': None,
                    'img_row_bounds': [3, 35],
                    'remove_stripe': {}},
            },
        }

        provenance = PipelineItem.get_data(
            data, schema='foxden.reader.FoxdenProvenanceReader')
        assert provenance == {
            'did': '/workflow=tomo_reconstruct',
            'input_files': [{'name': 'todo.fix: pipeline.yaml'}],
            'parent_did': None,
        }
