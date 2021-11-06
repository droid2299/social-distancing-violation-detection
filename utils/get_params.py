import argparse



def parse_arguments():

    parser = argparse.ArgumentParser(description='Social Distancing AI')

    parser.add_argument(
        '--input_video_path',
        type=str,
        default='data/input/input_video.mp4',
        help='''
            Path to the input video file to be processed''')

    parser.add_argument(
        '--output_video_path',
        type=str,
        default='data/output/output_T_big.mp4',
        help='''
            Path to which the output video file will be created
            Preferably go for .mp4 format as opposed to .avi
            to reduce the video file size''')

    parser.add_argument(
        '--report_file',
        type=str,
        default='data/output/report_T_b.txt',
        help='''
            Path to which the analytics should be written.''')
    parser.add_argument(
        '--od_model_path',
        type=str,
        default='object_detection/model/efficientdet_d4.pb',
        help='''
            Path to the Object Detection model
            Must be a frozen inference .pb model file''')

    parser.add_argument(
        '--od_label_map_path',
        type=str,
        default='object_detection/model/label_map.json',
        help='''
            Path to the json file containing mapping for all classes''')

    parser.add_argument(
        '--od_config_path',
        type=str,
        default='object_detection/model/config.json',
        help='''
            Path to the json file containing OD model configurations''')

    parser.add_argument(
        '--deepsort_model_path',
        type=str,
        default='object_tracking/deep_sort/model/mars-small128.pb',
        help='''
            Path to the deepsort model (trained on MARS dataset)
            Must be a frozen inference .pb model file''')

    parser.add_argument(
        '--scaling_factor',
        type=float,
        default=0.5,
        help='''
            Percentage by which to scale the height & width of the input video''')

    parser.add_argument(
        '--max_cosine_distance',
        type=float,
        default=0.3,
        help='''
            Distance threshold for associating tracks & detections in Deep Sort algorithm''')

    parser.add_argument(
        '--nn_budget',
        type=float,
        default=None,
        help='''
            The number of past frames to store deep appearance features''')

    return parser.parse_args()
