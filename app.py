import cv2
import numpy as np
import pandas as pd
import time
from collections import Counter
import itertools
import math

from utils import visualize
from utils import get_params
from utils import projection
from utils import violation_check

from object_tracking.deep_sort import nn_matching
from object_tracking.deep_sort.tracker import Tracker
from object_tracking.deep_sort.detection import Detection
from object_tracking.tools import generate_detections as gdet

from object_detection.efficientdet import EfficientNet
#import cProfile
#import re
import time


t_D=[]
t_T=[]
t_SD=[]
t_M=[]

def main():
    t_f1 = time.time()
    # Load EfficientNet for object detection
    object_detection = EfficientNet(
        FLAGS.od_model_path,
        FLAGS.od_label_map_path,
        FLAGS.od_config_path)

    # Load DeepSort for object tracking
    encoder = gdet.create_box_encoder(FLAGS.deepsort_model_path, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine', FLAGS.max_cosine_distance, FLAGS.nn_budget)
    tracker = Tracker(metric)

    # Calculate the Homography matrix for Perspective Transformation
    # homography, distance_threshold = projection.get_homography(
    #     FLAGS.input_video_path, FLAGS.scaling_factor)
    homography, upper_distance_threshold, lower_distance_threshold = projection.get_homography(
                                      FLAGS.input_video_path, FLAGS.scaling_factor)
    print(upper_distance_threshold,lower_distance_threshold)
    #homography = projection.get_homography(
    #    FLAGS.input_video_path, FLAGS.scaling_factor)

    video_capture = cv2.VideoCapture(FLAGS.input_video_path)
    video_writer = cv2.VideoWriter(
        FLAGS.output_video_path,
        cv2.VideoWriter_fourcc(*'MP4V'),
        video_capture.get(cv2.CAP_PROP_FPS), (1200,600))

    #df = pd.DataFrame(columns=["x1_b", "y1_b", "x2_b", "y2_b", "x_feet_b", "y_feet_b","x1", "y1", "x2", "y2","x_feet", "y_feet", "velocity", "id","frame_no"])
    df = pd.DataFrame(columns=["x1_b", "y1_b", "x2_b", "y2_b", "x_feet_b", "y_feet_b","id","frame_no"])
    group = pd.DataFrame(columns = ["person1", "person2", "grouped", "frame_start", "frame_end"])
    df_person_frame_count = pd.DataFrame(columns = ["PersonID", "Frame_entered", "Frame_exit"])

    frame_no = 0
    #variables for report generation
    unsafe_total_count=0
    unsafe_ids=[]
    unsafe_pairs=[]
    violation_id_counts_res=''
    Total_people_ids=[]
    total_people_count=0
    family_pairs=[]
    t1 = time.time()
    while True:
    #while frame_no < 10:
        ret, original_image = video_capture.read()
        if not ret:
            break
        print("frame number is: ",frame_no)
        original_image =  cv2.resize(original_image, dsize=(0, 0), fx=FLAGS.scaling_factor, fy=FLAGS.scaling_factor)
        cv2.imwrite('original.jpg' , original_image)
        t_d1 = time.time()
        detections, _ = object_detection.predict([original_image])
        print(detections)
        t_d2 = time.time()
        #print("detection time", t_d2-t_d1)
        t_D.append(t_d2-t_d1)
        t_t1 = time.time()
        features = encoder(original_image, detections)
        detections = [
            Detection(detection, 1.0, feature) 
            for detection, feature in zip(detections, features)]
        #t_d2 = time.time()
        #print("detection time", t_d2-t_d1)
        
        tracker.predict()
        tracker.update(detections)
        t_t2 = time.time()
        #print("tracker time", t_t2-t_t1)
        t_T.append(t_t2-t_t1)
        t_sd1 = time.time()
        track_data = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            #print(bbox)
            velocity=track.get_velocity()

            print("bounding_boxes", bbox)
            print("velocity", list(velocity))
            feet_point = ((bbox[0]) + (bbox[2] / 2), bbox[3])
            feet_point = np.array(feet_point, dtype=np.float32)[np.newaxis, :]
            transformed_feet_point = projection.transform_coords(feet_point, homography)

            track_data.append(
               (track.track_id, {
                   'feet_point': transformed_feet_point, 'velocity': velocity}))
            t_tr = time.time()
            #print("tracker time", t_tr-t_d2)


            #code for new social distance check
            x1,y1 = int(round(bbox[0])), int(round(bbox[1]))
            x2,y2 = int(round(bbox[2])), int(round(bbox[3]))
            xPos = bbox[2] + (bbox[0]-bbox[2])/2
            yPos = bbox[1] + (bbox[3]-bbox[1])/2
            x_feet, y_feet = int(round(xPos)), int(round(yPos))
            x1_b, y1_b = projection.transform_coords(np.array((x1,y1), dtype=np.float32)[np.newaxis, :], homography)
            x1_b, y1_b = int(round(x1_b)), int(round(y1_b))
            x2_b, y2_b = projection.transform_coords(np.array((x2,y2), dtype=np.float32)[np.newaxis, :], homography)
            x2_b, y2_b = int(round(x2_b)), int(round(y2_b))
            x_feet_b, y_feet_b = projection.transform_coords(np.array((x_feet,y_feet), dtype=np.float32)[np.newaxis, :], homography)
            x_feet_b, y_feet_b = int(round(x_feet_b)), int(round(y_feet_b))
            cv2.rectangle(original_image, (x1,y1), (x2, y2), (0,255,255), 2)
            #blurring the image
            try:
                shape = list(original_image[y1:y1+int((y2-y1)/4), x1:x2].shape)
                #print(shape)
                if (0 in shape):
                    pass
                else:
                    original_image[y1:y1+int((y2-y1)/4),x1:x2] = cv2.blur(original_image[y1:y1+int((y2-y1)/4), x1:x2], (10,10))
            except:
                pass


            lst = []
            lst.append(x1_b)
            lst.append(y1_b)
            lst.append(x2_b)
            lst.append(y2_b)
            lst.append(x_feet_b)
            lst.append(y_feet_b)
            #lst.append(x1)
            #lst.append(y1)
            #lst.append(x2)
            #lst.append(y2)
            #lst.append(x_feet)
            #lst.append(y_feet)
            #lst.append(list(velocity))
            lst.append(track.track_id)
            lst.append(frame_no)
            df = df.append(pd.DataFrame([lst], columns=df.columns), ignore_index=True)


        track_ids_status, track_connections, df,df_person_frame_count, group  = violation_check.social_distance_check(track_data,upper_distance_threshold, lower_distance_threshold, frame_no, df, df_person_frame_count, group, FLAGS.scaling_factor)
        t_sd2 = time.time()
        #print("sd logic time", t_sd2-t_sd1)
        t_SD.append(t_sd2-t_sd1)
        t_m1 = time.time()
        #code for report generation
        sdviolations_frame=[]

        for key,value in track_ids_status.items():
            if key not in Total_people_ids:
                total_people_count+=1
                Total_people_ids.append(key)

        for key,value in track_ids_status.items():
                if value=='unsafe':
                    #print(key,value)
                    sdviolations_frame.append(key)
                    if key not in unsafe_ids:
                        unsafe_total_count+=1
                        unsafe_ids.append(key)

        #print('Violations ids in this frame: ',sdviolations_frame)
        #print('Violations count this frame: ',len(sdviolations_frame))
        #print('Total cumulative violations count:',unsafe_total_count)
        #('All violation ids:',unsafe_ids)


        for key,value in track_connections.items():
            if value=='family':
                if key not in family_pairs:
                    family_pairs.append(key)
            elif value=='unsafe':
                if key not in unsafe_pairs:
                    unsafe_pairs.append(key)


        violation_idcounts = list(itertools.chain(*unsafe_pairs))
        violation_id_counts_res=dict(Counter(violation_idcounts))
        #print('All violation ids counts:',violation_id_counts_res)
        violation_id_counts_resframe = {k: violation_id_counts_res[k] for k in sdviolations_frame if k in violation_id_counts_res}
        #print('violation ids counts in this frame:',violation_id_counts_resframe)

        #code to get family count wrt ids
        for key,value in track_connections.items():
            if value=='family':
                if key not in family_pairs:
                    family_pairs.append(key)

        #code for report generation ended.        
        
        
        t_sd = time.time()
        #print("social distance check time", t_sd-t_tr)
        bird_view_image = np.zeros(shape=(1000, 1000, 3))
        original_image, bird_view_image = visualize.show_violations(
            original_image,
            bird_view_image,
            tracker.tracks,
            dict(track_data),
            track_ids_status,
            track_connections
        )

        #bird_view_image = cv2.resize(bird_view_image, (600, 600))
        original_image = cv2.resize(original_image, (1200, 600))
        #cv2.imwrite(str(frame_no)+'.jpg' , original_image)
        #print(original_image)
        #output_frame = np.concatenate((bird_view_image, original_image), axis=1)
        #output_frame = np.uint8(output_frame)
        t_m2 = time.time()
        #print("misc time", t_m2-t_m1)
        t_M.append(t_m2-t_m1)
        #cv2.imshow('Social Distancing AI', output_frame)
        video_writer.write(original_image)
        frame_no+=1
        t_f2 = time.time()
        #print("whole time", t_f2-t_f1)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    



    #fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # pd.DataFrame(violation_id_counts_res.items(),columns=['id','count']).to_csv('violation_id_counts_res.csv',index=False)
    # print('Total number of family/friends groups:',len(family_pairs))
    # #Frequencat10sec = unsafe_total_count/(fps*10)
    # #print('Frequency', Frequencat10sec)
    # print("family ids:", family_pairs)
    # print("Total violation count",unsafe_total_count)
    # print('Total People Count: ',total_people_count)
    # print("detection time is:", sum(t_D))
    # print("tracking time is:", sum(t_T))
    # print("sd time is :", sum(t_SD))
    # print("misc time:", sum(t_M)) 

    t2 = time.time()

    #with open(FLAGS.report_file, 'w') as f:
        #print('Total number of family/friends groups:', len(family_pairs), file=f)  # Python 3.x
        #print("family IDs:", family_pairs, file=f)
        #print("Total violation count",unsafe_total_count, file=f)
        #print('Total People Count: ',total_people_count, file=f)
        #print("detection time is:", sum(t_D), file=f)
        #print("tracking time is:", sum(t_T), file=f)
        #print("sd time is :", sum(t_SD), file=f)
        #print("misc time:", sum(t_M), file=f) 
        #print("total_time", t2-t1, file=f)

    video_capture.release()
    video_writer.release()
    #df.to_csv("./op.csv",index=False)
    #print("total_time", t2-t1)
    #cv2.destroyAllWindows()


if __name__ == '__main__':

    FLAGS = get_params.parse_arguments()
    main()
    #cProfile.run('re.compile("main()")')


