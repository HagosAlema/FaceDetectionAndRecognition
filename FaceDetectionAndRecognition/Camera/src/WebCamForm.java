
import com.googlecode.javacv.CanvasFrame;
import com.googlecode.javacv.FrameGrabber;
import com.googlecode.javacv.OpenCVFrameGrabber;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_core.cvFlip;
import com.googlecode.javacv.cpp.videoInputLib;

public class WebCamForm {

    public static void main(String[] args) {
        //Create canvas frame for displaying webcam.
        CanvasFrame canvas = new CanvasFrame("Webcam");

        //Set Canvas frame to close on exit
        canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);

        //Declare FrameGrabber to import output from webcam
        FrameGrabber grabber = new OpenCVFrameGrabber("");

        try {

      //Start grabber to capture video
            grabber.start();
            grabber.setImageHeight(640);
            grabber.setImageWidth(480);
            //Declare img as IplImage
            IplImage img;

            while (true) {

                //inser grabed video fram to IplImage img
                img = grabber.grab();
                

                //Set canvas size as per dimentions of video frame.
                canvas.setCanvasSize(grabber.getImageWidth(), grabber.getImageHeight());

                if (img != null) {
                    //Flip image horizontally
                    cvFlip(img, img, 1);
                    //Show video frame in canvas
                    canvas.showImage(img);
                }
            }
        } catch (Exception e) {
        }
    }
}
