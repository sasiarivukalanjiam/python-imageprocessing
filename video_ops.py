import numpy as np
import cv2

class VideoOps(object):

    def cam_view(self):
        """
        
        :return: 
        """
        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('From Webcam : press q to quit',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def read_video_file(self):
        """
        cap = cv2.VideoCapture('data/video/NFS.mp4')

        while (cap.isOpened()):
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Video : press q to quit', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        # cv2.error: C:\projects\opencv-python\opencv\modules\imgproc\src\color.cpp:11111: error: (-215) scn == 3 || scn == 4 in function cv::cvtColor
        """
        cap = cv2.VideoCapture('data/video/NFS.mp4')

        while(cap.isOpened()):  # check !
            # capture frame-by-frame
            ret, frame = cap.read()

            if ret: # check ! (some webcam's need a "warmup")
                # our operation on frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Display the resulting frame
                cv2.imshow('Video : press q to exit', gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything is done release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    vo = VideoOps()
    vo.cam_view()
    vo.read_video_file()
