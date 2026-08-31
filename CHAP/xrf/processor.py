#!/usr/bin/env python
"""Processors used only by XRF experiments."""

from pydantic import (
    Field,
    conlist,
)
from typing import (
    Optional,
)

from CHAP.processor import Processor
    
        
class PyMcaProcessor(Processor):
    """Processor to use [PyMca](https://github.com/vasole/pymca) to
    analyze XRF spectra. Reads in XRF spectra, incident beam flux, and
    a PyMca fit parameter configuration. Returns a list of
    dictionaries returned by the PyMca fit routine -- one dictionary
    for each XRF spectrum provided.
    """
    pipeline_fields: dict = Field(
        default={
            'pymca_config': 'dict',
        },
        init_var=True)
    pymca_config: dict = None

    def process(self, data):
        if self.pymca_config is None:
            self.pymca_config = self.get_data(
                data, schema='PyMca5.PyMcaPhysics.xrf.ClassMcaTheory')['data']

        advanced_fit, mass_fraction_tool = self.init_pymca()

        spectra = self.get_data(data, name='spectra')

        flux = self.get_data(data, name='flux')[:].tolist()

        results = self.analyze_spectra(
            advanced_fit, mass_fraction_tool,
            spectra, flux)

        return results

    def init_pymca(self):
        from PyMca5.PyMcaPhysics.xrf.ConcentrationsTool import (
            ConcentrationsTool)
        from PyMca5.PyMcaPhysics.xrf.ClassMcaTheory import McaTheory

        self.pymca_config['fit']['use_limit'] = 1
        advanced_fit = McaTheory(config=self.pymca_config)
        advanced_fit.enableOptimizedLinearFit()
        if 'concentrations' in self.pymca_config:
            mass_fraction_tool = ConcentrationsTool(
                self.pymca_config['concentrations']
            )
            mass_fraction_tool.config['time'] = 1
        else:
            mass_fraction_tool = None

        return advanced_fit, mass_fraction_tool
        
    def analyze_spectra(self, advanced_fit, mass_fraction_tool, spectra, flux):
        results = []
        npts = len(flux)
        self.logger.info(f'Analyzing {npts} spectra')
        for i, (s, f) in enumerate(zip(spectra, flux)):
            self.logger.debug(f'Analyzing spectrum {i}/{npts}')
            results.append(
                self.analyze_spectrum(
                    advanced_fit, mass_fraction_tool, s, f, i))
        return results

    def analyze_spectrum(
            self, advanced_fit, mass_fraction_tool, spectrum, flux, index):
        result = {}
        self.logger.debug(f'spectrum = {spectrum}')
        advanced_fit.setData(y=spectrum)
        advanced_fit.estimate()
        self.logger.debug(
            'Running PyMca5.PyMcaPhysics.xrf.ClassMcaTheory.McaTheory.startfit'
        )
        if (mass_fraction_tool is not None) \
           and (advanced_fit._fluoRates is None):
            fitresult, result = advanced_fit.startfit(digest=1)
        else:
            fitresult = advanced_fit.startfit(digest=0)
            result = advanced_fit.imagingDigestResult()
        self.logger.debug('Fit complete')

        if mass_fraction_tool is not None:
            self.logger.debug(
                'Using '
                'PyMca5.PyMcaPhysics.xrf.ConcentrationsTool.ConcentrationsTool'
                ' to get mass fractions'
            )
            temp = {}
            temp['fitresult'] = fitresult
            temp['result'] = result
            temp['result']['config'] = advanced_fit.config
            mass_fraction_tool.config['flux'] = flux
            conc = mass_fraction_tool.processFitResult(
                fitresult=temp,
                elementsfrommatrix=False,
                fluorates=advanced_fit._fluoRates
            )
            result['concentrations'] = conc

        return result


class PyMcaResultsProcessor(Processor):
    """Transforms a list of PyMca fit configuration result
    dictionaries (returned by :class:`~CHAP.xrf.processor.PyMcaProcessor`)
    into a basic `NXentry` structure.
    """
    def process(self, data):
        import numpy as np
        from nexusformat.nexus import NXdata, NXentry, NXfield

        self.logger.debug(f'len(data) = {len(data)}')
        for d in data:
            self.logger.debug(f'name {d["name"]} type {type(d["data"])}')
        results = self.get_data(data, name='PyMcaProcessor')
        elements = results[0]['concentrations']['groups']

        return NXentry(
            data=NXdata(
                **{
                    e.replace(' ', '_'): np.asarray(
                        [
                            r['concentrations']['mass fraction'][e]
                            for r in results
                        ],
                        dtype='float32',
                    ) for e in elements
                }
            )
        )


class NXfluoProcessor(Processor):
    """When provided with a list of
    :class:`~CHAP.pipeline.PipelineData` that have the appropriate
    `name` fields, this Processor uses those data to construct a (not
    yet) standardized representation of fluorescence data in NeXus
    format.
    """
    def process(self, data):
        from nexusformat.nexus import (
            NXdata,
            NXdetector,
            NXfield,
            NXentry,
            NXinstrument,
            NXlinkfield,
            NXmonitor,
            NXmonochromator,
            NXroot,
            NXsample,
            NXsource,
        )

        fluorescence = self.get_data(data, name='fluorescence')
        energy = self.get_data(data, name='energy')
        monitor = self.get_data(data, name='monitor')

        r = NXroot(
            entry=NXentry(
                title='???',
                start_time='??? ISO8601',
                definition='NXfluo',
                instrument=NXinstrument(
                    source=NXsource(
                        name='CHESS',
                        probe='x-ray',
                        **{'type': 'Synchrotron X-ray Source'}
                    ),
                    monochromator=NXmonochromator(
                        wavelength=NXfield(
                            value='??? float',
                            attrs={'units': '??? angstrom or nm'}
                        )
                    ),
                    fluorescence=NXdetector(
                        data=fluorescence,
                        energy=energy,
                    ),
                ),
                sample=NXsample(
                    name='???'
                ),
                monitor=NXmonitor(
                    mode='??? monitor or timer',
                    preset='??? float preset value for time or monitor idk what this means',
                    data=monitor
                ),
                data=NXdata(
                    energy=NXlinkfield(
                        target='/entry/instrument/fluorescence/energy'
                    ),
                    data=NXlinkfield(
                        target='/entry/instrument/fluorescence/data'
                    )
                )
            )
        )
        self.logger.debug(r.tree)

        return r


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
