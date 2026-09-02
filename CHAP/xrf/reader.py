#!/usr/bin/env python
"""XRF Readers"""


from CHAP.reader import Reader


class PyMcaFitConfigReader(Reader):
    """A file reader for PyMca5 fit parameter configuration (a
    `PyMca5.PyMcaIO.ConfigDict` object).

    If run with `interactive == `:
      - `True`: loads an existing parameter set from a file into the
         PyMca5 fit parameter configuration GUI. User may edit
         parameters in the GUI, optionally save the configuration to a
         new file, then click "OK" to return their edited
         configuration for later use by `CHAP.xrf.processor` tools.
      - `False`: loads an existing parameter set from a file and
         returns it for later use by `CHAP.xrf.processor` tools.
    """
    def read(self):
        from PyMca5.PyMcaIO.ConfigDict import ConfigDict

        from CHAP.pipeline import PipelineData

        config = ConfigDict()
        config.read(self.filename)

        if self.interactive:
            from PyMca5.PyMcaGui import PyMcaQt as qt
            from PyMca5.PyMcaGui.physics.xrf.FitParam import FitParamDialog
            app = qt.QApplication([])
            self.logger.debug('before fpd construction')
            fpd = FitParamDialog(modal=1)
            self.logger.debug('after fpd construction')
            fpd.loadParameters(self.filename)
            self.logger.debug(f'after fpd.loadParameters({self.filename})')
            config = fpd.getParameters()
            self.logger.debug('after fpd.getParameters')
            ret = fpd.exec()
            if ret == qt.QDialog.Accepted:
                config = fpd.getParameters()
                del fpd

        return PipelineData(
            data=config, schema='PyMca5.PyMcaPhysics.xrf.ClassMcaTheory')

if __name__ == '__main__':
    # Local modules
    from CHAP.reader import main

    main()
