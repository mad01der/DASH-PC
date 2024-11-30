/*2024/11/26 author:JinyangLi */
Following the next steps:
1.install the nginx server 
2.the mpd and source files in folder date is what you need to put in nginx html,origin contains the original date while source contains the sample date.
3.to start the project, you need to run "npm run dev",as the project is built on vite.
4.there are two available mpd files : one is the pointcloud.mpd while other is pointcloud_origin.mpd. the former is used to show the improvement of the video system,the later is used to try the origin dateset.
however,the results are not so good because the origin ply files are very huge.So the next improvement is to try draco.

recommend website:https://blog.csdn.net/jxt1234and2010/article/details

/*2024/11/30 */
add draco in draco_test.js ,the draco files can be found in public/drc, to try the draco ,change the index source file from "main.js" to "draco_test.js".
According to the result of my experiment, the browser can load max point number is around 30,000. Video quality is just bearable under this situation.
                                                                                                        
