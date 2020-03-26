import argparse
from os import getenv, makedirs, environ
from os.path import splitext, basename, exists, isfile, join
import numpy as np
import yaml
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.tools import *
from utils.box_utils import decode, decode_landm
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from tqdm import tqdm
from utils.tiktok import tik_tok

with open('env.yml', 'r') as file:
    load = yaml.safe_load(file)
    for key, value in load.items():
        environ[key] = value


def arg_parse():
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('-m', '--trained_model',
                        default=getenv('resnet'),
                        type=str,
                        help='Trained state_dict file path to open')

    parser.add_argument('--network',
                        default='resnet50',
                        help='Backbone network mobile0.25 or resnet50')

    parser.add_argument('--cpu',
                        action="store_true",
                        default=False,
                        help='Use cpu inference')

    parser.add_argument('--confidence_threshold',
                        default=float(getenv('confidence_threshold')),
                        type=float,
                        help='confidence_threshold')

    parser.add_argument('--top_k',
                        default=int(getenv('top_k')),
                        type=int,
                        help='top_k')

    parser.add_argument('--nms_threshold',
                        default=float(getenv('nms_threshold')),
                        type=float,
                        help='nms_threshold')

    parser.add_argument('--step',
                        default=10,
                        type=int,
                        help='speed up the process')

    parser.add_argument('--keep_top_k',
                        default=int(getenv('keep_top_k')),
                        type=int,
                        help='keep_top_k')

    parser.add_argument('--vis_thres',
                        default=float(getenv('vis_thres')),
                        type=float,
                        help='visualization_threshold')

    parser.add_argument('--input',
                        default='upload/test.mp4',
                        type=str,
                        help='video input for process')

    parser.add_argument('--output-dir',
                        default=getenv('output_dir'))

    args = parser.parse_args()
    return args


@tik_tok
def pipeline(net, frame, args, device, resize, cfg):
    img = np.float32(frame)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]
    dets = np.concatenate((dets, landms), axis=1)

    objects_to_draw = dict(draw_box=True, draw_text=True, draw_landmarks=True)
    frame = draw(frame, dets, args.vis_thres, **objects_to_draw)
    return frame


def main():
    args = arg_parse()

    filename, extension = splitext(basename(args.input))
    print("Loading file [{}] ....".format(filename))

    if not exists(args.input):
        raise ValueError("File [{}] is not recognized".format(args.input))

    if not isfile(args.trained_model):
        raise ValueError(f'The model {args.trained_model} is not found')

    if not exists(args.output_dir):
        print(f'Output directory {args.output_dir} does not exist, Creating one')
        makedirs(args.output_dir)

    torch.set_grad_enabled(False)
    cfg = cfg_mnet if args.network == "mobile0.25" else cfg_re50
    resize = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model_cfg(args.trained_model, cfg=cfg, device=device, cpu=args.cpu)

    if is_video(extension):
        vdo = cv2.VideoCapture()
        codec = cv2.VideoWriter_fourcc(*'XVID')
        output = join(args.output_dir, filename + '.avi')

        if vdo.open(args.input):
            property_id = int(cv2.CAP_PROP_FRAME_COUNT)
            total_frames = int(cv2.VideoCapture.get(vdo, property_id))
            width = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vdo.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(output, codec, fps, (width, height))

            print('')
            print('processing video ...')
            frame_idx = 0
            with tqdm(range(total_frames)) as pbar:
                while vdo.grab():
                    frame_idx += 1
                    pbar.update(1)
                    ret, frame = vdo.retrieve()

                    if not ret:
                        break

                    args.step = 1 if args.step < 1 else args.step
                    if frame_idx % args.step == 0:
                        frame = pipeline(net, frame, args, device, resize, cfg)
                        writer.write(frame)

            pbar.close()
            print('process finished Successfully')
            print('process finished. file is stored as {}'.format(output))

            vdo.release()
            writer.release()

    elif is_image(extension):
        frame = cv2.imread(args.input)
        img = np.float32(frame)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf, landms = net(img)  # forward pass
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]
        dets = np.concatenate((dets, landms), axis=1)

        objects_to_draw = dict(draw_box=True, draw_text=False, draw_landmarks=False)
        frame = draw(frame, dets, args.vis_thres, **objects_to_draw)

        output = args.output_dir + filename + '.jpg'
        cv2.imwrite(output, frame)
        print('output is stored as {}'.format(output))

    else:
        print('cant read video {}'.format(args.input))


if __name__ == '__main__':
    main()
