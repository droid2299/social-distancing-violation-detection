import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance


def distance_check(point1, point2, distance_threshold):

    # Calculate the Euclidean distance between the feet points of the 2 tracker boxes
    euclid_distance = np.linalg.norm(point1 - point2)

    return 'higher' if euclid_distance > distance_threshold else 'lower'


def angle_check(velocity_1, velocity_2, angle_threshold):
    # Compute the unit vectors for the 2 velocity vectors
    velocity_1_unit = velocity_1 / np.linalg.norm(velocity_1)
    velocity_2_unit = velocity_2 / np.linalg.norm(velocity_2)

    # Compute the angle between the 2 unit vectors
    angle = np.degrees(np.arccos(np.dot(velocity_1_unit, velocity_2_unit)))

    return 'higher' if angle > angle_threshold else 'lower'


def velocity_check(velocity_1, velocity_2, velocity_threshold):
    # Compute the magnitudes of the 2 velocity vectors
    velocity_1_magnitude = np.linalg.norm(velocity_1)
    velocity_2_magnitude = np.linalg.norm(velocity_2)

    # Compute the percentage within which the magnitudes are
    # If x=10 and y=12, then percent_range is calculated as follows
    # |(x - y) ÷ min(x, y)| * 100 %
    percent_range = 100 * abs(
        (velocity_1_magnitude - velocity_2_magnitude) /
        min(velocity_1_magnitude, velocity_2_magnitude))

    return 'higher' if percent_range > velocity_threshold else 'lower'


def calculate_euclidean_distance(p1, p2):
    dist = distance.euclidean(p1, p2)
    return dist


def distance_check_new(frame_no, df, df_person_frame_count, group, scaling_factor, lower_distance_threshold):
    df_temp = df[df.frame_no==frame_no]
    x_b_centre = df_temp["x_feet_b"].tolist()
    y_b_centre = df_temp["y_feet_b"].tolist()
    #x_centre = df_temp["x_feet"].tolist()
    #y_centre = df_temp["y_feet"].tolist()
    person_id = [int(x) for x in df_temp["id"].tolist()]
    
    person_id_allframes_unique = df_person_frame_count["PersonID"].tolist()
    #print("df_temp", df_temp)
    #print("df_person_frame_count", df_person_frame_count)

    for per_id in person_id:
        #if person in present in total persons list 
        if per_id in person_id_allframes_unique:
            index = df_person_frame_count.index[df_person_frame_count['PersonID'] == per_id].tolist()
            df_person_frame_count.loc[index,"Frame_exit"] = int(frame_no)
        # if new person is spotted in frame
        else:
            df_person_frame_count = df_person_frame_count.append(pd.Series([int(per_id), int(frame_no), int(frame_no)], index=['PersonID','Frame_entered', 'Frame_exit']), ignore_index=True)
    
    #print("df_person_frame_count", df_person_frame_count)
    for xb1,yb1,p1 in zip(x_b_centre, y_b_centre, person_id):
        for xb2, yb2,p2 in zip(x_b_centre, y_b_centre, person_id):
            if p1<p2:
                #distance calculation in birds eye view
                dist = calculate_euclidean_distance((xb1,yb1), (xb2,yb2))
                if dist<lower_distance_threshold: #scaling_factor:
                    #image_np[int(y1):int(y1)+10, int(x1):int(x1)+10] = (255,0,0)
                    #image_np[int(y2):int(y2)+10, int(x2):int(x2)+10] = (255,0,0)
                    #group = group.append([frame, [p1,p2]])
                    #group = group.append(pd.Series([frame, str(p1)+"-"+str(p2)], index=['frame_no','group']), ignore_index=True)
                    group_ids = group["grouped"].tolist() 
                    #print("group_ids", group_ids)
                    #print("group", group)
                    #print("group", group)
                    #print("RHS", group_ids)
                    #print("LHS", str(p1)+"-"+str(p2))
                    if(str(p1)+"-"+str(p2) in group_ids):
                        #print("1")
                        #str(p1)+"-"+str(p2)
                        #('{}-{}').format(p1,p2)
                        index = group.grouped[group.grouped == str(p1)+"-"+str(p2)].index.tolist()
                        #print(index)
                        group.loc[index,"frame_end"] = frame_no
                    else:
                        #print("2")
                        group = group.append(pd.Series([p1,p2,str(p1)+"-"+str(p2),frame_no], index=["person1", "person2","grouped", "frame_start"]), ignore_index=True)
    
    #print("group", group)
    group_1 = group[["person1", "grouped", "frame_start", "frame_end"]]
    group_1.rename(columns={'person1': 'PersonID'}, inplace=True)
    group_2 = group[["person2", "grouped", "frame_start", "frame_end"]]
    group_2.rename(columns={'person2': 'PersonID'}, inplace=True)
    group_new = pd.concat([group_1, group_2], ignore_index=True)
    group_new = group_new.sort_values("PersonID")
    group_new["frame_diff_group"] = group_new["frame_end"]-group_new["frame_start"]

    #df_person_frame_count = df_person_frame_count[df_person_frame_count['PersonID'].notna()]
    #df_person_frame_count = df_person_frame_count.astype(int)
    df_person_frame_count["frame_diff_person"] = df_person_frame_count["Frame_exit"] - df_person_frame_count["Frame_entered"]

    
    df_final = pd.merge(group_new, df_person_frame_count, on='PersonID')
    if frame_no == 0:
        pass
    else:
        try:
            df_final["group_ratio"] = df_final["frame_diff_group"]/df_final["frame_diff_person"]
        except:
            df_final["group_ratio"] = 0
            pass
        df_final.drop(["frame_start", "frame_end", "Frame_entered","Frame_exit"],axis=1, inplace=True)
        df_group = df_final[df_final["group_ratio"] > 0.8]
        df_group.rename(columns={'PersonID': 'id'}, inplace=True)
        #print(df_group)
        info_group_person=pd.merge(df_temp, df_group, on="id")
    return df, df_person_frame_count, group, info_group_person


def social_distance_check(track_data, upper_distance_threshold, lower_distance_threshold, frame_no, df, df_person_frame_count, group, scaling_factor):

    track_ids_status = {}
    track_connections = {}

    for i in range(len(track_data) - 1):
        track_1_id = track_data[i][0]

        for j in range(i + 1, len(track_data)):
            track_2_id = track_data[j][0]
            #print("id pairs", [track_1_id, track_2_id])

            # If that person has never been encountered, initialize them as safe
            if track_1_id not in track_ids_status:
                track_ids_status[track_1_id] = 'safe'
            if track_2_id not in track_ids_status:
                track_ids_status[track_2_id] = 'safe'
            track_connections[(track_1_id, track_2_id)] = 'safe'

            # Elaborate Checks if the 2 persons are within the social distance threshold
            if distance_check(track_data[i][1]['feet_point'], track_data[j][1]['feet_point'], upper_distance_threshold) == 'lower':

                #track_ids_status[track_1_id] = 'unsafe'
                #track_ids_status[track_2_id] = 'unsafe'
                #track_connections[(track_1_id, track_2_id)] = 'unsafe'
                
                # If the angles of movement are above the threshold
                # They are strangers walking in different directions
                if angle_check(track_data[i][1]['velocity'], track_data[j][1]['velocity'], 30) == 'higher':
                    track_ids_status[track_1_id] = 'unsafe'
                    track_ids_status[track_2_id] = 'unsafe'
                    track_connections[(track_1_id, track_2_id)] = 'unsafe'

                else:

                    # If the distance betweetrack_data[i][1]['velocity'], track_data[j][1]['velocity']n the 2 persons are greater than the family distance threshold
                    # Then they are strangers, walking moderately close and in the same direction
                    if distance_check(track_data[i][1]['feet_point'], track_data[j][1]['feet_point'], lower_distance_threshold) == 'higher':
                        track_ids_status[track_1_id] = 'unsafe'
                        track_ids_status[track_2_id] = 'unsafe'
                        track_connections[(track_1_id, track_2_id)] = 'unsafe'

                    else:

                        # If the velocity is greater than the velocity percent threshold
                        # Then they are strangers, walking quickly past each other, close and in same direction
                        if velocity_check(track_data[i][1]['velocity'], track_data[j][1]['velocity'], 40) == 'higher':
                            track_ids_status[track_1_id] = 'unsafe'
                            track_ids_status[track_2_id] = 'unsafe'
                            track_connections[(track_1_id, track_2_id)] = 'unsafe'
                        else:
                            df, df_person_frame_count, group, info_group_person = distance_check_new(frame_no, df, df_person_frame_count, group, scaling_factor, lower_distance_threshold)
                            ids = info_group_person["id"].tolist()
                            #print("info_df", info_group_person)
                            #print("all ids in frame", ids)
                            #print("track1", track_1_id)
                            #print("track2", track_2_id)

                            # track_ids_status[track_1_id] = 'family'
                            # track_ids_status[track_2_id] = 'family'
                            # track_connections[(track_1_id, track_2_id)] = 'family'
                            
                            if ((track_1_id in ids) and (track_2_id in ids)):
                                track_ids_status[track_1_id] = 'family'
                                track_ids_status[track_2_id] = 'family'
                                track_connections[(track_1_id, track_2_id)] = 'family'
                            else:
                                track_ids_status[track_1_id] = 'unsafe'
                                track_ids_status[track_2_id] = 'unsafe'
                                track_connections[(track_1_id, track_2_id)] = 'unsafe'

    return track_ids_status, track_connections, df, df_person_frame_count, group
