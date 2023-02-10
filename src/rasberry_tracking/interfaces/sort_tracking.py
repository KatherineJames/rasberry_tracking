#!/usr/bin/env python3

# Provides re-id features of bounding box detections
#  Raymond Kirk (Tunstill) Copyright (c) 2021
#  Email: ray.tunstill@gmail.com
#
# This file aims to interface trackers/detectors with github.com/RaymondKirk/rasberry_perception
from __future__ import absolute_import, division, print_function

from rasberry_perception.interfaces.fruitcast import FruitCastServer
from rasberry_perception.interfaces.registry import DETECTION_REGISTRY
from rasberry_perception.utility import function_timer

from rasberry_perception.msg import Detections
from rasberry_perception.visualisation import Visualiser
from sensor_msgs.msg import Image
from std_msgs.msg import String

from scipy.spatial import distance

import rospy
import ros_numpy
import numpy as np
from filterpy.kalman import KalmanFilter
from rospy.exceptions import ROSInterruptException


# SORT code adapted from : https://github.com/abewley/sort
def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def distance_batch(bb_test, bb_gt):
  """
  Computes the euclidean distance between two bboxes,each in the form [x1,y1,x2,y2]
  """  
  xt1 = bb_test[..., 0]
  yt1 = bb_test[..., 1]
  xt2 = bb_test[..., 2]
  yt2 = bb_test[..., 3] 

  xg1 = bb_gt[..., 0]
  yg1 = bb_gt[..., 1]
  xg2 = bb_gt[..., 2]
  yg2 = bb_gt[..., 3]  

  # find centre of each box
  w_test = np.maximum(0., xt2 - xt1)
  h_test = np.maximum(0., yt2 - yt1) 
  x_test = bb_test[..., 0] + w_test/2.
  y_test = bb_test[..., 0] + h_test/2.

  w_gt = np.maximum(0., xg2 - xg1)
  h_gt = np.maximum(0., yg2 - yg1)
  x_gt = bb_gt[..., 0] + w_gt/2.
  y_gt = bb_gt[..., 0] + h_gt/2.

  # create list of points to compare
  p_test = np.column_stack((x_test,y_test))
  p_gt = np.column_stack((x_gt,y_gt))  
  # print("test",p_test.shape[0])
  # print("gt",p_gt.shape[0])

  # compute the distance between the centres
  D = distance.cdist(p_test,p_gt)
  return D

def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """ 
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])  
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  # print(w)

 
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3, dist_threshold = 30):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_compare = False  
  
  if iou_compare:
    iou_matrix = iou_batch(detections, trackers) 
    if min(iou_matrix.shape) > 0:
      a = (iou_matrix > iou_threshold).astype(np.int32) # get the indices where there is a match
      # print(multiple)
      if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
      else:
        matched_indices = linear_assignment(-iou_matrix)
        # print(matched_indices)
    else:
      matched_indices = np.empty(shape=(0,2))

  else: # use euclidean distance between centres
    dist_matrix = distance_batch(detections,trackers)    
    argmin = np.argmin(dist_matrix,axis=1)   
    a = np.zeros(dist_matrix.shape)   
    for i in range(0,a.shape[0]):   
      if dist_matrix[i,argmin[i]] < dist_threshold:
        a[i,argmin[i]]  = 1
    
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-dist_matrix)      
  
  

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  
  matches = []
  for m in matched_indices:
    if iou_compare and (iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))


@DETECTION_REGISTRY.register_detection_backend("SORTRasberryTracker")
class SORTRasberryTracker(): #FruitCastServer
    def __init__(self):
        # Custom code here to generate a tracker
        # raise NotImplementedError("Please write a custom tracking interface")       
        # Subscribe
        self.cam_sub = rospy.Subscriber( 
            "/rasberry_perception/results", Detections, self.get_detector_results)         

        # Publish results 
        self.tracked_dets_vis_pub = rospy.Publisher("rasberry_perception/"+ "tracking/track_boxes", Image, queue_size=1)

        self.object_counter = 0

        self.tracker = Sort()          
    

    def get_detector_results(self, msg):
        # raise NotImplementedError("Please write a custom tracking interface")       
        
        dets = []
        detections = msg.objects 
        
        for i in range(0,len(detections)):
            roi = detections[i].roi
            x1 = roi.x1
            x2 = roi.x2
            y1 = roi.y1
            y2 = roi.y2
            score = detections[i].confidence
            dets.append([x1,y1,x2,y2,score])

        if len(detections) == 0:
            dets = np.empty((0, 5))
        
        updated = self.tracker.update(dets = np.array(dets))

        image_msg = msg.camera_frame
        vis = Visualiser(ros_numpy.numpify(image_msg))

        for i in range(0,len(updated)):
            x1,y1,x2,y2,obj_id = updated[i]
            vis.draw_box([x1, y1, x2, y2], (1,0,0)) 
            vis.draw_text(f"{obj_id}",[x1,y1], font_scale=0.5)  

        vis_image = vis.get_image(overlay_alpha=0.5)
        vis_msg = ros_numpy.msgify(Image, vis_image, encoding=image_msg.encoding)
        # vis_msg.header = image_msg.header
        # vis_info = image_info
        # vis_info.header = vis_msg.header
        self.tracked_dets_vis_pub.publish(vis_msg)
           
def __tracking_runner():
    # Command line arguments should always over ride ros parameters
    # service_name = default_service_name
    # _node_name = service_name + "_server"
    
    _node_name = "tracking"
    rospy.init_node(_node_name)
    
    tracker = SORTRasberryTracker()  
    rospy.spin()    

if __name__ == '__main__': 
    __tracking_runner()
    
    
    

"""
Look at the maskrcnn interface for help
- read in the detections (bboxes)
- apply sort algorithm to bboxes

Tracking must publish these topics:
/rasberry_perception/tracking/detection_marker_array
/rasberry_perception/tracking/detection_results
/rasberry_perception/tracking/detection_results_array
/rasberry_perception/tracking/marker_array
/rasberry_perception/tracking/pose
/rasberry_perception/tracking/pose_array
/rasberry_perception/tracking/results
/rasberry_perception/tracking/results_array
"""