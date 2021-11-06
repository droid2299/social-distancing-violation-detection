import cv2
import numpy as np


def draw_plane(frame, points, x, y, color):

    for t in np.linspace(0, 1, 5):
        v_1 = (int((points[0][0] * t) + (points[1][0] * (1 - t))), int((points[0][1] * t) + (points[1][1] * (1 - t))))
        v_2 = (int((x * t) + (points[2][0] * (1 - t))), int((y * t) + (points[2][1] * (1 - t))))
        h_1 = (int((points[0][0] * t) + (x * (1 - t))), int((points[0][1] * t) + (y * (1 - t))))
        h_2 = (int((points[1][0] * t) + (points[2][0] * (1 - t))), int((points[1][1] * t) + (points[2][1] * (1 - t))))
        cv2.line(frame, v_1, v_2, color, 2)
        cv2.line(frame, h_1, h_2, color, 2)

    return frame


def interactive_gui(image):
    '''
    Launch a GUI which allows user to mark 6 points on the image,
    the first four points correspond to the points which lie in a plane,
    the next 2 points correspond to 6ft distance in the original image

    Parameters
    -----------
    image - three dimensional numpy array

    Returns
    ---------
    points - list
    '''
    color_yellow = (0, 255, 255)
    color_blue =  (255, 255, 0)
    preview = None
    points = []

    def draw_tool(event, x, y, flags, param):
        nonlocal preview, image

        # Recording 4 points for Perspective Transformation
        if len(points) < 4:
            if event == cv2.EVENT_LBUTTONUP and len(points) == 0:
                points.append((x, y))
                cv2.circle(image, points[0], 5, color_blue, -1)
                preview = image.copy()

            elif event == cv2.EVENT_MOUSEMOVE and len(points) == 1:
                preview = image.copy()
                cv2.line(preview, points[0], (x, y), color_yellow, 2)

            elif event == cv2.EVENT_LBUTTONUP and len(points) == 1:
                points.append((x, y))
                cv2.circle(image, points[1], 5, color_blue, -1)
                cv2.line(image, points[0], points[1], color_yellow, 2)
                preview = image.copy()

            elif event == cv2.EVENT_MOUSEMOVE and len(points) == 2:
                preview = image.copy()
                for t in np.linspace(0, 1, 5):
                    v_1 = (int((points[0][0] * t) + (points[1][0] * (1 - t))), int((points[0][1] * t) + (points[1][1] * (1 - t))))
                    h_1 = (int((points[0][0] * t) + (x * (1 - t))), int((points[0][1] * t) + (y * (1 - t))))
                    h_2 = (int((points[1][0] * t) + (x * (1 - t))), int((points[1][1] * t) + (y * (1 - t))))
                    cv2.line(preview, v_1, (x, y), color_yellow, 2)
                    cv2.line(preview, h_1, h_2, color_yellow, 2)

            elif event == cv2.EVENT_LBUTTONUP and len(points) == 2:
                points.append((x, y))
                cv2.circle(image, points[2], 5, color_blue, -1)
                cv2.line(image, points[1], points[2], color_yellow, 2)
                preview = image.copy()

            elif event == cv2.EVENT_MOUSEMOVE and len(points) == 3:
                preview = image.copy()
                cv2.line(preview, points[2], (x, y), color_yellow, 2)
                preview = draw_plane(preview, points, x, y, color_yellow)

            elif event == cv2.EVENT_LBUTTONUP and len(points) == 3:
                points.append((x, y))
                cv2.circle(image, points[3], 5, color_blue, -1)
                image = draw_plane(image, points, x, y, color_yellow)
                preview = None
        elif len(points) < 8:
            if event == cv2.EVENT_LBUTTONUP and len(points) == 4:
                points.append((x, y))
                cv2.circle(image, points[4], 5, color_blue, -1)
                preview = image.copy()

            elif event == cv2.EVENT_MOUSEMOVE and len(points) == 5:
                preview = image.copy()
                cv2.line(preview, points[4], (x, y), color_yellow, 2)

            elif event == cv2.EVENT_LBUTTONUP and len(points) == 5:
                points.append((x, y))
                cv2.circle(image, points[5], 5, color_blue, -1)
                cv2.line(image, points[4], points[5], color_yellow, 2)
                preview = image.copy()

            elif event == cv2.EVENT_MOUSEMOVE and len(points) == 6:
                preview = image.copy()
                for t in np.linspace(0, 1, 5):
                    v_1 = (int((points[4][4] * t) + (points[5][4] * (1 - t))), int((points[4][5] * t) + (points[5][5] * (1 - t))))
                    h_1 = (int((points[4][4] * t) + (x * (1 - t))), int((points[4][5] * t) + (y * (1 - t))))
                    h_2 = (int((points[5][4] * t) + (x * (1 - t))), int((points[5][5] * t) + (y * (1 - t))))
                    cv2.line(preview, v_1, (x, y), color_yellow, 2)
                    cv2.line(preview, h_1, h_2, color_yellow, 2)

            elif event == cv2.EVENT_LBUTTONUP and len(points) == 6:
                points.append((x, y))
                cv2.circle(image, points[6], 5, color_blue, -1)
                cv2.line(image, points[5], points[6], color_yellow, 2)
                preview = image.copy()

            elif event == cv2.EVENT_MOUSEMOVE and len(points) == 7:
                preview = image.copy()
                cv2.line(preview, points[6], (x, y), color_yellow, 2)
                preview = draw_plane(preview, points[4:7], x, y, color_yellow)

            elif event == cv2.EVENT_LBUTTONUP and len(points) == 7:
                points.append((x, y))
                cv2.circle(image, points[7], 5, color_blue, -1)
                image = draw_plane(image, points[4:7], x, y, color_yellow)
                preview = None

        # Recording 2 points for social-distance threshold
        elif len(points) < 10:
            if event == cv2.EVENT_LBUTTONUP and len(points) == 8:
                points.append((x, y))
                cv2.circle(image, points[8], 5, color_blue, -1)
                preview = image.copy()

            elif event == cv2.EVENT_MOUSEMOVE and len(points) == 9:
                preview = image.copy()
                cv2.line(preview, points[8], (x, y), color_yellow, 2)

            elif event == cv2.EVENT_LBUTTONUP and len(points) == 9:
                points.append((x, y))
                preview = None
                cv2.line(image, points[8], points[9], color_yellow, 2)
                cv2.circle(image, points[9], 5, color_blue, -1)

        else:
            if event == cv2.EVENT_LBUTTONUP and len(points) == 10:
                points.append((x, y))
                cv2.circle(image, points[10], 5, color_blue, -1)
                preview = image.copy()

            elif event == cv2.EVENT_MOUSEMOVE and len(points) == 11:
                preview = image.copy()
                cv2.line(preview, points[10], (x, y), color_yellow, 2)

            elif event == cv2.EVENT_LBUTTONUP and len(points) == 11:
                points.append((x, y))
                preview = None
                cv2.line(image, points[10], points[11], color_yellow, 2)
                cv2.circle(image, points[11], 5, color_blue, -1)

    # Set Mouse Callback
    cv2.namedWindow(winname='Social Distancing GUI') 
    cv2.setMouseCallback('Social Distancing GUI', draw_tool)
    
    # show the image
    while len(points) < 12:
        if preview is None:
            cv2.imshow('Social Distancing GUI', image)
        else:
            cv2.imshow('Social Distancing GUI', preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(points)
    cv2.destroyAllWindows()
    return points