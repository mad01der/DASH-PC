/*2024/11/26*/
Following the next steps:
1.install the nginx server 
2.the mpd and source files in folder date is what you need to put in nginx html,origin contains the original date while source contains the sample date.
3.to start the project, you need to run "npm run dev",as the project is built on vite.
4.there are two available mpd files : one is the pointcloud.mpd while other is pointcloud_origin.mpd. the former is used to show the improvement of the video system,the later is used to try the origin dateset.

/*2024/11/30*/
add draco in draco_test.js ,the draco files can be found in public/drc, to try the draco ,change the index source file from "main.js" to "draco_test.js".
According to the result of my experiment, the browser can load max point number is around 30,000. Video quality is just bearable in this situation.

/*2024/12/13*/
add play_control.js and the layout of the page has been re-optimized.

                                                                                                        
