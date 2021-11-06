import cv2


def draw_status(original_image, bird_view_image, tracks, track_data, track_ids_status):
    if track_ids_status == {}:
        pass
    else:
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            id = track.track_id
            bbox = track.to_tlbr()
            bbox = list(map(int, bbox))    

            if track_ids_status[id] == 'unsafe':
                color = (0, 0, 255)  # Red
            elif track_ids_status[id] == 'family':
                color = (0, 0, 255)  # Blue
            elif track_ids_status[id] == 'safe':
                color = (0, 255, 0)  # Green
            else:
                color = (0,255,255)

        # Visualization on Bird View Image
            cv2.circle(bird_view_image,
                (int(track_data[id]['feet_point'][0]), int(track_data[id]['feet_point'][1])),
                5, color, -1)
            cv2.putText(bird_view_image, str(id),
                (int(track_data[id]['feet_point'][0]), int(track_data[id]['feet_point'][1] + 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

        # Visualization on Video Feed
            cv2.rectangle(original_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    return original_image, bird_view_image


def draw_connections(bird_view_image, track_data, track_connections):

    for id_pair, status in track_connections.items():
        point1 = tuple(map(int, track_data[id_pair[0]]['feet_point']))
        point2 = tuple(map(int, track_data[id_pair[1]]['feet_point']))

        if status == 'unsafe':
            color = (0, 0, 255)  # Red
        elif status == 'family':
            color = (255, 0, 0)  # Blue
        elif status == 'safe':
            continue  # No connecting lines if they are at safe distance
        
        # Visualization on Bird View Image
        cv2.line(bird_view_image, point1, point2, color, thickness=2)

    return bird_view_image


def show_violations(original_image, bird_view_image, tracks, track_data, track_ids_status, track_connections):

    original_image, bird_view_image = draw_status(original_image, bird_view_image, tracks, track_data, track_ids_status)
    bird_view_image = draw_connections(bird_view_image, track_data, track_connections)

    return original_image, bird_view_image
