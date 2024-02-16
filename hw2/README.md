## 1c - Combination criteria and thresholds
For matching the objects in the image with the database, I used the roundness.
1. For the labelled image, I use 1b to obtain a object database
2. Then I compare this object database with the one provided in the function argument. 
3. I compare for each object the roundness and see if the absolute difference is within 0.02. I came to this threshold after observing the difference in the roundness of the same objects across the different images. 
4. If the roundness satisfies this condition, I plot the position and orientation of the object.
