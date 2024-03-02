# 766
Assignments for Computer Vision Class CS 766
1b. Used standard voting scheme 
1c. Although I used a threshold value, I ran a suppression algorithm before threshold to suppress all small hough bins around a large one
1d. The logic here is very convoluted, however I have commented the code well, please take a look there. In a gist, I used the edge_img to figure out if the current pixel is on an edge or not. If it on the edge, I connect it to the previous pixel that was on an edge, provided its not "corrupted". By "corrupted" here I mean that the previous pixel was on the edge but the current is not and so. 
