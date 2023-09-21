import torch
import modules.scripts as scripts
import gradio as gr
from modules.script_callbacks import on_cfg_denoiser
from modules import processing
from torchvision import transforms
import numpy as np

class Script(scripts.Script):

    def title(self):
        return "Latent Mirroring extension"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Latent Mirroring", open=False):
                mirror_mode = gr.Radio(label='Latent Mirror mode', choices=['None', 'Alternate Steps', 'Blend Average', 'Impose'], value='None', type="index")
                mirror_style = gr.Radio(label='Latent Mirror style', choices=['Horizontal Mirroring', 'Vertical Mirroring', 'Horizontal+Vertical Mirroring', '90 Degree Rotation', '180 Degree Rotation', 'Roll Channels', 'None'], value='Horizontal Mirroring', type="index")

                with gr.Row():
                    x_pan = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, label='X panning', value=0.0)
                    y_pan = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, label='Y panning', value=0.0)

                mirroring_fractions = gr.Textbox(label='List of what fractions (of iterations) to symmetrise in', value='0.1')

                if not is_img2img:
                    disable_hr = gr.Checkbox(label='Disable during hires pass', value=False)
                else:
                    disable_hr = gr.State(False)

        self.run_callback = False
        return [mirror_mode, mirror_style, x_pan, y_pan, mirroring_fractions, disable_hr]

    def denoise_callback(self, params):
        is_hires = self.is_hires

        # indices start at -1
        # params.sampling_step = max(0, real_sampling_step)
        if params.sampling_step >= params.total_sampling_steps - 2:
            self.is_hires = not is_hires and self.enable_hr

        if not self.run_callback or is_hires:
            return

        #if params.sampling_step >= params.total_sampling_steps * self.mirroring_max_step_fraction:
         #   return
        mirroring_its = np.round(self.mirroring_fractions).astype(int)
        if params.sampling_step not in mirroring_its:
            return

        try:
            if self.mirror_mode == 1:
                if self.mirror_style == 0:
                    params.x[:, :, :, :] = torch.flip(params.x, [3])
                elif self.mirror_style == 1:
                    params.x[:, :, :, :] = torch.flip(params.x, [2])
                elif self.mirror_style == 2:
                    params.x[:, :, :, :] = torch.flip(params.x, [3, 2])
                elif self.mirror_style == 3:
                    params.x[:, :, :, :] = torch.rot90(params.x, dims=[2, 3])
                elif self.mirror_style == 4:
                    params.x[:, :, :, :] = torch.rot90(torch.rot90(params.x, dims=[2, 3]), dims=[2, 3])
                elif self.mirror_style == 5:
                    params.x[:, :, :, :] = torch.roll(params.x, shifts=1, dims=[1])

            elif self.mirror_mode == 2:
                if self.mirror_style == 0:
                    params.x[:, :, :, :] = (torch.flip(params.x, [3]) + params.x)/2
                elif self.mirror_style == 1:
                    params.x[:, :, :, :] = (torch.flip(params.x, [2]) + params.x)/2
                elif self.mirror_style == 2:
                    params.x[:, :, :, :] = (torch.flip(params.x, [2, 3]) + params.x)/2
                elif self.mirror_style == 3:
                    params.x[:, :, :, :] = (torch.rot90(params.x, dims=[2, 3]) + params.x)/2
                elif self.mirror_style == 4:
                    params.x[:, :, :, :] = (torch.rot90(torch.rot90(params.x, dims=[2, 3]), dims=[2, 3]) + params.x)/2
                elif self.mirror_style == 5:
                    params.x[:, :, :, :] = (torch.roll(params.x, shifts=1, dims=[1]) + params.x)/2

            
            elif self.mirror_mode == 3:
                input_image = params.x
                if self.mirror_style == 0:
                    left_half = input_image[:, :, :, :input_image.size(3) // 2]
                    right_half = torch.flip(left_half, [3])
                    params.x[:, :, :, :] = torch.cat((left_half, right_half), dim=3)
                elif self.mirror_style == 1:
                    top_half = input_image[:, :, :input_image.size(2) // 2, :]
                    bottom_half = torch.flip(top_half, [2])
                    params.x[:, :, :, :] = torch.cat((top_half, bottom_half), dim=2)
                elif self.mirror_style == 2:
                    height, width = input_image.size(2), input_image.size(3)
                    top_left = input_image[:, :, :height // 2, :width // 2]
                    top_right = torch.flip(top_left, [3])  # Flip horizontally
                    bottom_left = torch.flip(top_left, [2])  # Flip vertically
                    bottom_right = torch.flip(bottom_left, [3])  # Flip horizontally and vertically
                    top_half = torch.cat((top_left, top_right), dim=3)
                    bottom_half = torch.cat((bottom_left, bottom_right), dim=3)
                    params.x[:, :, :, :] =  torch.cat((top_half, bottom_half), dim=2)
                elif self.mirror_style == 3:
                    height, width = input_image.size(2), input_image.size(3)
                    top_left = input_image[:, :, :height // 2, :width // 2]
                    rotated_top_left = torch.rot90(top_left, k=1, dims=(2, 3))
                    top_right = torch.flip(rotated_top_left, [2])  # Flip horizontally
                    bottom_left = torch.flip(rotated_top_left, [3])  # Flip vertically
                    bottom_right = torch.flip(bottom_left, [2])  # Flip horizontally
                    top_half = torch.cat((rotated_top_left, top_right), dim=3)
                    bottom_half = torch.cat((bottom_left, bottom_right), dim=3)
                    params.x[:, :, :, :] = torch.cat((top_half, bottom_half), dim=2)
                elif self.mirror_style == 4:
                    height, width = input_image.size(2), input_image.size(3)
                    top_half = input_image[:, :, :height // 2, :]
                    rotated_top_half = torch.flip(top_half, [2, 3])  # Flip both vertically and horizontally
                    params.x[:, :, :, :] = torch.cat((rotated_top_half, input_image[:, :, height // 2:, :]), dim=2)
                elif self.mirror_style == 5:
                    params.x[:, :, :, :] = (torch.roll(params.x, shifts=1, dims=[1]) + params.x)/2
                    
        except RuntimeError as e:
            if self.mirror_style in (3, 4):
                raise RuntimeError('90 Degree Rotation requires a square image.') from e
            else:
                raise RuntimeError('Error transforming image for latent mirroring.') from e

        if self.x_pan != 0:
            params.x[:, :, :, :] = torch.roll(params.x, shifts=int(params.x.size()[3]*self.x_pan), dims=[3])
        if self.y_pan != 0:
             params.x[:, :, :, :] = torch.roll(params.x, shifts=int(params.x.size()[2]*self.y_pan), dims=[2])


    def process(self, p, mirror_mode, mirror_style, x_pan, y_pan, mirroring_fractions, disable_hr):
        self.mirror_mode = mirror_mode
        self.mirror_style = mirror_style
        self.mirroring_fractions = np.fromstring(mirroring_fractions, dtype=float, sep=',')
        self.x_pan = x_pan
        self.y_pan = y_pan

        if mirror_mode != 0:
            p.extra_generation_params["Mirror Mode"] = mirror_mode
            p.extra_generation_params["Mirror Style"] = mirror_style
            p.extra_generation_params["Mirroring Fractions"] = np.fromstring(mirroring_fractions, dtype=float, sep=',')
        if x_pan != 0:
            p.extra_generation_params["X Pan"] = x_pan
        if y_pan != 0:
            p.extra_generation_params["Y Pan"] = y_pan

        if not hasattr(self, 'callbacks_added'):
            on_cfg_denoiser(self.denoise_callback)
            self.callbacks_added = True
        self.run_callback = True
        self.enable_hr = getattr(p, 'enable_hr', False) and not disable_hr
        self.is_hires = False

    def postprocess(self, *args):
        self.run_callback = False
        return
