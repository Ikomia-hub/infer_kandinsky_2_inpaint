import copy
from ikomia import core, dataprocess
import torch
import numpy as np
import random
from diffusers import AutoPipelineForInpainting
import os
from PIL import Image
import cv2


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferKandinsky2InpaintParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.prompt = "a god, high quality"
        self.prior_guidance_scale = 4.0
        self.guidance_scale = 1.0
        self.negative_prompt = "lowres, text, error, cropped, worst quality, low quality, ugly"
        self.prior_num_inference_steps = 25
        self.num_inference_steps = 150
        self.seed = -1
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.prompt = str(param_map["prompt"])
        self.guidance_scale = float(param_map["guidance_scale"])
        self.prior_guidance_scale = float(param_map["prior_guidance_scale"])
        self.negative_prompt = str(param_map["negative_prompt"])
        self.seed = int(param_map["seed"])
        self.num_inference_steps = int(param_map["num_inference_steps"])
        self.prior_num_inference_steps = int(param_map["prior_num_inference_steps"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["prompt"] = str(self.prompt)
        param_map["negative_prompt"] = str(self.negative_prompt)
        param_map["guidance_scale"] = str(self.guidance_scale)
        param_map["prior_guidance_scale"] = str(self.prior_guidance_scale)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        param_map["prior_num_inference_steps"] = str(self.prior_num_inference_steps)
        param_map["seed"] = str(self.seed)

        return param_map


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferKandinsky2Inpaint(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.add_input(dataprocess.CInstanceSegmentationIO())
        self.add_input(dataprocess.CSemanticSegmentationIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferKandinsky2InpaintParam())
        else:
            self.set_param_object(copy.deepcopy(param))
        
        self.device = torch.device("cpu")
        self.pipe = None
        self.bin_img = None
        self.generator = None
        self.seed = None
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
        self.model_name = "kandinsky-community/kandinsky-2-2-decoder-inpaint"

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def load_model(self, param, local_files_only):
        torch_tensor_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            self.model_name,
            torch_dtype=torch_tensor_dtype,
            use_safetensors=True,
            cache_dir=self.model_folder,
            local_files_only=local_files_only
            )

        self.pipe.to(self.device)
        param.update = False

    def generate_seed(self, seed):
        if seed == -1:
            self.seed = random.randint(0, 191965535)
        else:
            self.seed = seed
        self.generator = torch.Generator(self.device).manual_seed(self.seed)

    def convert_and_resize_img(self, scr_image):
        img = Image.fromarray(scr_image)
        # Stride of 128
        new_width = 128 * (scr_image.shape[1] // 128)
        new_height = 128 * (scr_image.shape[0] // 128)

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        print('IMAAAGE', new_width, new_height)
        return resized_img, new_height, new_width

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get image input
        img_input = self.get_input(0).get_image()

        init_image, height, width = self.convert_and_resize_img(img_input)

        # init_image = Image.fromarray(img_input)

        # Get parameters
        param = self.get_param_object()

        # Load pipeline
        if param.update or self.pipe is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            try:
                self.load_model(param, local_files_only=True)
            except Exception as e:
                self.load_model(param, local_files_only=False)

            self.generate_seed(param.seed)
        
        # Get mask or create it from graphics input
        inst_input = self.get_input(2)  # Instance segmentation mask
        seg_input = self.get_input(3)  # Semantic segmentation mask
        if inst_input.is_data_available():
            print("Instance segmentation mask available")
            self.bin_img = inst_input.get_merge_mask()
        elif seg_input.is_data_available():
            print("Semantic segmentation mask available")
            self.bin_img = seg_input.get_mask()
        else:
            if img_input.dtype == 'uint8':
                imagef = np.asarray(img_input, dtype=np.float32)/255
                graph_input = self.get_input(1)
                if graph_input.is_data_available():
                    print("Graphics input available")
                    self.create_graphics_mask(
                        imagef.shape[1], imagef.shape[0], graph_input)
                    self.bin_img = self.get_graphics_mask(0)

            else:
                raise NotImplementedError("Image of data type {} not supported".format(img_input.dtype))


        if self.bin_img is not None:
            mask_image = cv2.resize(self.bin_img, (height, width))

        else:
            raise Exception("No graphic input set.")
        

        with torch.no_grad():
            result = self.pipe(prompt=param.prompt,
                        negative_prompt=param.negative_prompt,
                        image=init_image,
                        mask_image=mask_image,
                        prior_num_inference_steps=param.prior_num_inference_steps,
                        prior_guidance_scale=param.prior_guidance_scale,
                        guidance_scale=param.guidance_scale,
                        height=height,
                        width=width,
                        generator=self.generator,
                        num_inference_steps=param.num_inference_steps,
                        ).images[0]

        # Get and display output
        image = np.array(result)
        output_img = self.get_output(0)
        output_img.set_image(image)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferKandinsky2InpaintFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_kandinsky_2_inpaint"
        self.info.short_description = "Kandinsky 2.2 inpainting diffusion model."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Diffusion"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/einstein.jpg"
        self.info.authors = "A. Shakhmatov, A. Razzhigaev, A. Nikolich, V. Arkhipkin, I. Pavlov, A. Kuznetsov, D. Dimitrov"
        self.info.article = "https://aclanthology.org/2023.emnlp-demo.25/"
        self.info.journal = "ACL Anthology"
        self.info.year = 2023
        self.info.license = "Apache 2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder"
        # Code source repository
        self.info.repository = "https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint"
        self.info.original_repository = "https://github.com/ai-forever/Kandinsky-2"
        # Keywords used for search
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "INPAINTING"
        self.info.keywords = "Latent Diffusion,Hugging Face,Kandinsky,Inpaint,Generative"

    def create(self, param=None):
        # Create algorithm object
        return InferKandinsky2Inpaint(self.info.name, param)
