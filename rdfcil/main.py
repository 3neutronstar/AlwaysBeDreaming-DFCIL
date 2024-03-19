# -*- coding: utf-8 -*-

from cl_lite.core import App

from datamodule import DataModule
from module import Module
# from iscf_module_ijcv import ISCFModule
# from iscf_module import ISCFModule
# from iscf_module_eccv_supplementary2 import ISCFModule
# from iscf_module_eccv_supplementary import ISCFModule
# from iscf_module_ft_balanced import ISCFModuleFTB
from iscf_module_ft import ISCFModuleFT


app = App(
    # ISCFModuleFTB,
    ISCFModuleFT,
    # ISCFModule,
    # Module,
    DataModule,
    gpus=-1,
    benchmark=True,
    num_sanity_val_steps=0,
    replace_sampler_ddp=False,
    deterministic=True,
    checkpoint_callback=False,
    check_val_every_n_epoch=40,
    max_epochs=200,
)

if __name__ == "__main__":
    app.main()
