import os
import math
import time
import glob
import cv2
import numpy as np
import scipy.ndimage
import torch

from argparse import ArgumentParser

# Set deterministic mode for reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

GOP_8 = [[1, 9, 5], [1, 5, 3], [1, 3, 2], [3, 5, 4], [5, 9, 7], [5, 7, 6], [7, 9, 8]]

GOP_16 = [[1, 17, 9], [1, 9, 5], [1, 5, 3], [1, 3, 2], [3, 5, 4], [5, 9, 7], [5, 7, 6], [7, 9, 8], [9, 17, 13],
          [9, 13, 11], [9, 11, 10], [11, 13, 12], [13, 17, 15], [13, 15, 14], [15, 17, 16]]

GOP_32 = [[1, 33, 17],
          [1, 17, 9], [1, 9, 5], [1, 5, 3], [1, 3, 2], [3, 5, 4], [5, 9, 7], [5, 7, 6], [7, 9, 8], [9, 17, 13],
          [9, 13, 11], [9, 11, 10], [11, 13, 12], [13, 17, 15], [13, 15, 14], [15, 17, 16],
          [17, 33, 25], [17, 25, 21], [17, 21, 19], [17, 19, 18], [19, 21, 20], [21, 25, 23], [21, 23, 22],
          [23, 25, 24], [25, 33, 29],
          [25, 29, 27], [25, 27, 26], [27, 29, 28], [29, 33, 31], [29, 31, 30], [31, 33, 32]]


def mse2psnr(mse):
    if mse > 0:
        return 10 * (torch.log(1 * 1 / mse) / np.log(10))
    else:
        return 100


def calculate_psnr_rgb(x_hat, x):
    mse = torch.mean((x_hat - x).pow(2))
    psnr = mse2psnr(mse)
    return psnr


def validate_gop(batch, checkpoint_file, n_gop=32):
    model = Compressor(None).cuda()
    checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    batch = torch.unsqueeze(torch.stack(batch).permute(1, 0, 2, 3), 0)

    # Prepare buffer for decoded frame
    buffer = [None] * n_gop
    total_rate = [None] * n_gop
    total_psnr = [None] * n_gop

    # Intra coding for first frame
    print("Encoding Frame #1")
    x = batch[:, :, 0, :, :]
    x_hat, intra_rate = model.intra_coding(x.cuda())
    x_hat = torch.clamp(x_hat, 0, 1).detach().cpu()
    x_hat_psnr = calculate_psnr_rgb(x_hat, x)

    buffer[0] = x_hat
    total_rate[0] = (intra_rate, 0, 0)
    total_psnr[0] = (x_hat_psnr)

    print("Decoded Frame #1: PSNR: {}, BPP: {}".format(x_hat_psnr, intra_rate))

    # Intra coding for last frame
    print("Encoding Frame #{}".format(n_gop))
    x = batch[:, :, n_gop - 1, :, :]
    x_hat, intra_rate = model.intra_coding(x.cuda())
    x_hat = torch.clamp(x_hat, 0, 1).detach().cpu()
    x_hat_psnr = calculate_psnr_rgb(x_hat, x)

    buffer[n_gop - 1] = x_hat
    total_rate[n_gop - 1] = (intra_rate, 0, 0)
    total_psnr[n_gop - 1] = (x_hat_psnr)

    print("Decoded Frame #{}: PSNR: {}, BPP: {}".format(n_gop, x_hat_psnr, intra_rate))

    if n_gop == 9:
        used_gop = GOP_8
    elif n_gop == 17:
        used_gop = GOP_16
    elif n_gop == 33:
        used_gop = GOP_32
    else:
        raise ValueError(f"Invalid n_gop value of {n_gop}.")

    for id, gop in enumerate(used_gop):

        if abs(gop[0] - gop[1]) <= 4:
            reference = False
        else:
            reference = True

        print(f"Encoding Frame #{gop[2]}")
        x = batch[:, :, (gop[2] - 1), :, :]

        x_hat, base_rate, enhancement_rate = model.inter_coding(buffer[gop[0] - 1].cuda(), x.cuda(),
                                                                buffer[gop[1] - 1].cuda(), reference)

        x_hat = torch.clamp(x_hat, 0, 1).detach().cpu()
        x_hat_psnr = calculate_psnr_rgb(x_hat, x)
        total_psnr[gop[2] - 1] = x_hat_psnr
        total_rate[gop[2] - 1] = (0, base_rate, enhancement_rate)

        buffer[gop[2] - 1] = x_hat

        print(
            f"Decoded Frame #{gop[2]}: PSNR: {x_hat_psnr}, BPP: {base_rate + enhancement_rate} (BL: {base_rate}, EL: {enhancement_rate})")

    return buffer, total_psnr, total_rate


class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = int(self.width * self.height * 3 / 2)
        self.f = open(filename, 'rb')
        self.shape = (int(self.height * 1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            return False, None, True
        return True, yuv, False

    def read(self):
        ret, yuv, end_gop = self.read_raw()
        if end_gop:
            return ret, yuv, end_gop
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
        return ret, bgr, end_gop

    def read_yuv420(self):
        ret, yuv, end_gop = self.read_raw()
        if end_gop:
            return ret, yuv, end_gop

        yuv_shape = yuv.shape
        y_size = int(yuv_shape[0] * 2 / 3)
        uv_size = int(yuv_shape[0] * 1 / 6)

        y_channel = yuv[0:y_size, :]
        u_channel = yuv[y_size:(y_size + uv_size), :int(yuv_shape[1] / 2)]
        v_channel = yuv[(y_size + uv_size):, :int(yuv_shape[1] / 2)]

        u_channel = scipy.ndimage.zoom(u_channel, (4, 2))
        v_channel = scipy.ndimage.zoom(v_channel, (4, 2))

        yuv_channel = np.stack((y_channel, u_channel, v_channel), axis=2)

        return ret, yuv_channel, end_gop


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    g_gop = int(args.group_gop)
    n_gop = int(args.gop)

    with open("{0}/{1}".format(args.save_dir, "logs.txt"), "a") as f:
        f.write(
            "Filename, PSNR-RGB (dB), Intra Rate (bpp), Base Rate (bpp), Enhancement Rate (bpp), Total Rate (bpp)\n")

    for i, filepath in enumerate(glob.iglob(args.dataset_dir + "/*.yuv")):
        if filepath.endswith(".yuv") and i > -1:
            size = (int(args.height), int(args.width))
            cap = VideoCaptureYUV(filepath, size)
            gop_frames = []
            gop_counter = 0
            global_counter = 0

            print(filepath)

            dirname = os.path.basename(filepath)[:-4]
            directory = os.path.join(args.save_dir, dirname)
            os.makedirs(directory, exist_ok=True)

            while 1:
                if global_counter == g_gop:
                    break

                ret, frame, end_gop = cap.read()
                if ret or end_gop:

                    if not end_gop:
                        frame = frame / 255
                        frame = np.transpose(frame, (2, 0, 1))
                        frame = torch.from_numpy(frame)
                        frame = frame.type(torch.float32)

                        gop_frames.append(frame)
                        gop_counter += 1

                    # each gop should be multiplication of 8
                    if (len(gop_frames) == (n_gop + 1) or end_gop) and (len(gop_frames) - 1) % 8 == 0:
                        frames_reconstruction, frames_psnr, bitrates = validate_gop(gop_frames, args.load,
                                                                                    len(gop_frames))

                        # write log file

                        for frame_index in range(len(gop_frames) - 1):
                            filename = "{}_{:04d}".format(os.path.basename(filepath)[:-4],
                                                          (global_counter * n_gop) + frame_index + 1)

                            x_hat = frames_reconstruction[frame_index]
                            intra_rate, base_rate, enhancement_rate = bitrates[frame_index]
                            total_rate = intra_rate + base_rate + enhancement_rate

                            x_hat_psnr = frames_psnr[frame_index]

                            with open("{0}/{1}".format(args.save_dir, "logs.txt"), "a") as f:
                                f.write(
                                    "{}, {:0.3f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}\n".format(
                                        filename, x_hat_psnr, intra_rate, base_rate, enhancement_rate, total_rate))

                        global_counter += 1
                        gop_counter = 1
                        gop_frames = [gop_frames[n_gop]]

                        print('Batch #{0}/{1}'.format(global_counter, g_gop))

                    if end_gop:
                        break
                else:
                    break
                gop_counter += 1

            continue
        else:
            continue


if __name__ == '__main__':
    parser = ArgumentParser(add_help=True)

    # training specific
    parser.add_argument('dataset_dir')
    parser.add_argument('model_name')
    parser.add_argument('load')
    parser.add_argument('--save_dir', default='logs/' + str(int(time.time())))
    parser.add_argument('--gop', default=32)
    parser.add_argument('--group_gop', default=3)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)

    # parse params
    args, unknown = parser.parse_known_args()

    if args.model_name == 'tlzmc-plus':
        from tlzmc_plus import Compressor
    elif args.model_name == 'tlzmc-dstar':
        from tlzmc_dstar import Compressor
    elif args.model_name == 'tlzmc-star':
        from tlzmc_star import Compressor
    else:
        raise ValueError("Invalid model_name: {}".format(model_name))

    main(args)
