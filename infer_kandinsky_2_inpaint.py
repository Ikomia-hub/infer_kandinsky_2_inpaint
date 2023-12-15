from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_kandinsky_2_inpaint.infer_kandinsky_2_inpaint_process import InferKandinsky2InpaintFactory
        return InferKandinsky2InpaintFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_kandinsky_2_inpaint.infer_kandinsky_2_inpaint_widget import InferKandinsky2InpaintWidgetFactory
        return InferKandinsky2InpaintWidgetFactory()
