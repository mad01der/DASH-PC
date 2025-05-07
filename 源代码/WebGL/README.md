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
                                                                                                        
/*2025/3/12 */
todo list for the next period:
1.Modify the VR button rules so that clicking the button directly triggers the playback function; 
2.Increase the frame rate from 60fps to 180fps; 
3.Add functionality to record VR headset data coordinates; determine if the headset's coordinates have changed to assess whether real displacement has occurred and the extent of the displacement;
4.Discuss with the senior to request a more spacious experimental area.

/*2025/3/14 */
todo list for the next period:
1.Adjust the proportions of the girl model;
2.Verify whether the surround simulation can be achieved;
3.Retrieve the parameters for rotation.
                                                                                                        
/*2025/3/15 */
论文里面可以写的算法包括：
1.带宽测量和带宽的变化幅度
2.显著性检测，依据显著性检测的结果进行分块,对于每一块进行不同程度的下采样
3.视口预测(粗略)
4.视口预测(精确)
论文里面可以介绍的系统：
DASH-PC   DASH-PC(VR support)
其中，算法第1点和DASH-PC可以作为基础的系统搭建；算法第2点和第三点可以作为传输优化的算法；算法第4点和DASH-PC(VR support)可以作为加入头显后的系统实际效果；记得把预测算法优化一下。
周记从4月份开始写，每周一，周三，周五完成一篇周记；

/*2025/3/16 */
经测量，使用linux系统的预测速度远远大于windows系统

/*2025/4/11 */
写论文前还需做的事情
1.增加登录页面和点播页面   --down
2.增加功能选择功能 -- down
3.增加WebXR功能  --down 待头显验证
4.修改FPS的计算功能 --down
5.扩充数据集至72帧 -- down
5.优化视口预测算法 --down
ddl:20th April