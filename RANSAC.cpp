int main()
{
//RANSAC

//load edge image
cv::Mat color = cv::imread("../circleDetectionEdges.png");

// convert to grayscale
cv::Mat gray;
cv::cvtColor(color, gray, CV_RGB2GRAY);

// get binary image
cv::Mat mask = gray > 0;
//erode the edges to obtain sharp/thin edges (undo the blur?)
cv::erode(mask, mask, cv::Mat());

std::vector<cv::Point2f> edgePositions;
edgePositions = getPointPositions(mask);

// create distance transform to efficiently evaluate distance to nearest edge
cv::Mat dt;
cv::distanceTransform(255-mask, dt,CV_DIST_L1, 3);

//TODO: maybe seed random variable for real random numbers.

unsigned int nIterations = 0;

char quitKey = 'q';
std::cout << "press " << quitKey << " to stop" << std::endl;
while(cv::waitKey(-1) != quitKey)
{
    //RANSAC: randomly choose 3 point and create a circle:
    //TODO: choose randomly but more intelligent,
    //so that it is more likely to choose three points of a circle.
    //For example if there are many small circles, it is unlikely to randomly choose 3 points of the same circle.
    unsigned int idx1 = rand()%edgePositions.size();
    unsigned int idx2 = rand()%edgePositions.size();
    unsigned int idx3 = rand()%edgePositions.size();

    // we need 3 different samples:
    if(idx1 == idx2) continue;
    if(idx1 == idx3) continue;
    if(idx3 == idx2) continue;

    // create circle from 3 points:
    cv::Point2f center; float radius;
    getCircle(edgePositions[idx1],edgePositions[idx2],edgePositions[idx3],center,radius);

    float minCirclePercentage = 0.4f;

    // inlier set unused at the moment but could be used to approximate a (more robust) circle from alle inlier
    std::vector<cv::Point2f> inlierSet;

    //verify or falsify the circle by inlier counting:
    float cPerc = verifyCircle(dt,center,radius, inlierSet);

    if(cPerc >= minCirclePercentage)
    {
        std::cout << "accepted circle with " << cPerc*100.0f << " % inlier" << std::endl;
        // first step would be to approximate the circle iteratively from ALL INLIER to obtain a better circle center
        // but that's a TODO

        std::cout << "circle: " << "center: " << center << " radius: " << radius << std::endl;
        cv::circle(color, center,radius, cv::Scalar(255,255,0),1);

        // accept circle => remove it from the edge list
        cv::circle(mask,center,radius,cv::Scalar(0),10);

        //update edge positions and distance transform
        edgePositions = getPointPositions(mask);
        cv::distanceTransform(255-mask, dt,CV_DIST_L1, 3);
    }

    cv::Mat tmp;
    mask.copyTo(tmp);

    // prevent cases where no fircle could be extracted (because three points collinear or sth.)
    // filter NaN values
    if((center.x == center.x)&&(center.y == center.y)&&(radius == radius))
    {
        cv::circle(tmp,center,radius,cv::Scalar(255));
    }
    else
    {
        std::cout << "circle illegal" << std::endl;
    }

    ++nIterations;
    cv::namedWindow("RANSAC"); cv::imshow("RANSAC", tmp);
}

std::cout << nIterations <<  " iterations performed" << std::endl;


cv::namedWindow("edges"); cv::imshow("edges", mask);
cv::namedWindow("color"); cv::imshow("color", color);

cv::imwrite("detectedCircles.png", color);
cv::waitKey(-1);
return 0;
}


float verifyCircle(cv::Mat dt, cv::Point2f center, float radius, std::vector<cv::Point2f> & inlierSet)
{
 unsigned int counter = 0;
 unsigned int inlier = 0;
 float minInlierDist = 2.0f;
 float maxInlierDistMax = 100.0f;
 float maxInlierDist = radius/25.0f;
 if(maxInlierDist<minInlierDist) maxInlierDist = minInlierDist;
 if(maxInlierDist>maxInlierDistMax) maxInlierDist = maxInlierDistMax;

 // choose samples along the circle and count inlier percentage
 for(float t =0; t<2*3.14159265359f; t+= 0.05f)
 {
     counter++;
     float cX = radius*cos(t) + center.x;
     float cY = radius*sin(t) + center.y;

     if(cX < dt.cols)
     if(cX >= 0)
     if(cY < dt.rows)
     if(cY >= 0)
     if(dt.at<float>(cY,cX) < maxInlierDist)
     {
        inlier++;
        inlierSet.push_back(cv::Point2f(cX,cY));
     }
 }

 return (float)inlier/float(counter);
}


inline void getCircle(cv::Point2f& p1,cv::Point2f& p2,cv::Point2f& p3, cv::Point2f& center, float& radius)
{
  float x1 = p1.x;
  float x2 = p2.x;
  float x3 = p3.x;

  float y1 = p1.y;
  float y2 = p2.y;
  float y3 = p3.y;

  // PLEASE CHECK FOR TYPOS IN THE FORMULA :)
  center.x = (x1*x1+y1*y1)*(y2-y3) + (x2*x2+y2*y2)*(y3-y1) + (x3*x3+y3*y3)*(y1-y2);
  center.x /= ( 2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2) );

  center.y = (x1*x1 + y1*y1)*(x3-x2) + (x2*x2+y2*y2)*(x1-x3) + (x3*x3 + y3*y3)*(x2-x1);
  center.y /= ( 2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2) );

  radius = sqrt((center.x-x1)*(center.x-x1) + (center.y-y1)*(center.y-y1));
}



std::vector<cv::Point2f> getPointPositions(cv::Mat binaryImage)
{
 std::vector<cv::Point2f> pointPositions;

 for(unsigned int y=0; y<binaryImage.rows; ++y)
 {
     //unsigned char* rowPtr = binaryImage.ptr<unsigned char>(y);
     for(unsigned int x=0; x<binaryImage.cols; ++x)
     {
         //if(rowPtr[x] > 0) pointPositions.push_back(cv::Point2i(x,y));
         if(binaryImage.at<unsigned char>(y,x) > 0) pointPositions.push_back(cv::Point2f(x,y));
     }
 }

 return pointPositions;
}
