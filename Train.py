import linknet

import json
import cv2 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

class Linknet():
	def __init__(self):
		self.Parser = argparse.ArgumentParser()
		self.Parser.add_argument("-i","--InitParameters",help="Json file containig the initialization parameters")
		self.Parser.add_argument("-t","--TrainDictionary",help="Json file containing the dictionary for training elements")
		self.Parser.add_argument("-x","--TestDictionary",help="Json file containing the dictionary for testing elements")
		self.Args=self.Parser.parse_args()
		
		with open(self.Args.InitParameters) as JsonFile:
			self.InitParameters=json.load(JsonFile)

		with open(self.Args.TrainDictionary) as JsonFile:
			self.TrainDictionary=json.load(JsonFile)
		

		with open(self.Args.TestDictionary) as JsonFile:
			self.TestDictionary=json.load(JsonFile)
		
		self.Inputs=tf.placeholder(tf.float32, shape=[None, self.InitParameters["Height"], self.InitParameters["Width"], 3])
		self.Model=linknet.building_block(inputs=self.Inputs,classes=self.InitParameters["NumClasses"])
		self.Outputs = tf.nn.softmax(self.Model)
		self.GroundTruths=tf.placeholder(tf.float32, shape=[None, self.InitParameters["Height"], self.InitParameters["Width"], self.InitParameters["NumClasses"]])
		self.Cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Model, labels=self.GroundTruths))
		self.Optimizer = tf.train.AdamOptimizer().minimize(self.Cost)
		self.TrainLossTracker,self.TestLossTracker=[],[]
		self.GpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
		self.Session = tf.Session()
		self.Saver=tf.train.Saver()
		self.Initialize= tf.initialize_all_variables()
		self.Session.run(self.Initialize)

		self.Train()

	def FetchBatch(self, mode=0):
		assert mode ==0 or mode ==1
		if mode == 0 :
			IterableList=list(self.TrainDictionary.keys())[:len(self.TrainDictionary.keys())-len(self.TrainDictionary.keys())%self.InitParameters["BatchSize"]]
		if mode == 1 :
			IterableList=list(self.TestDictionary.keys())[:len(self.TestDictionary.keys())-len(self.TestDictionary.keys())%self.InitParameters["BatchSize"]]
			
		
		IterableList= [IterableList[i:i+self.InitParameters["BatchSize"]] for i in range(0, len(IterableList), self.InitParameters["BatchSize"])]
		
		for Batch in IterableList:
			Inputs=[]
			GroundTruths=[]
			for InputName in Batch:
				print(InputName)
				Inputs.append(cv2.imread(InputName))
				if mode ==0:
					GroundTruths.append(np.load(self.TrainDictionary[InputName])) 
				if mode ==1:
					GroundTruths.append(np.load(self.TestDictionary[InputName]))
			yield np.array(Inputs),np.array(GroundTruths)		

	def UpdateConfusionMatrix(self,Outputs,GroundTruths):
		for TargetClass in range(self.InitParameters["NumClasses"]):
			self.ConfusionMatrix[TargetClass] *= self.CounterMatrix[TargetClass]
			Mask = GroundTruths[:,:,TargetClass]
			self.CounterMatrix[TargetClass]+=np.sum(Mask)
			for PredictedClass in range(self.InitParameters["NumClasses"]):
				self.ConfusionMatrix[TargetClass,PredictedClass]+=np.sum(Outputs[:,:,PredictedClass]*GroundTruths[:,:,TargetClass])
			self.ConfusionMatrix[TargetClass,:]/=self.CounterMatrix[TargetClass]	

	def GenerateMetrics(self,Epoch):
		TrainX=[x * self.InitParameters["BatchSize"]/len(self.TrainDictionary.keys()) for x in range(0, len(self.TrainLossTracker) )]
		TestX=[x * self.InitParameters["BatchSize"]/len(self.TestDictionary.keys()) for x in range(0, len(self.TestLossTracker))]

		plt.plot(TrainX,self.TrainLossTracker)
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.savefig("TrainingLosses.png")
		plt.close()

		plt.plot(TestX,self.TestLossTracker)
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.savefig("TestingLosses.png")
		plt.close()

		sns.set()
		ax = sns.heatmap(self.ConfusionMatrix)
		plt.savefig("Epoch-"+str(Epoch)+".png")


	def RunEpoch(self,ConfusionMatrix=None,mode=0,Epoch=1):
		Iterator = self.FetchBatch(mode=mode)
	
		for Steps,(Inputs,GroundTruths) in enumerate(Iterator):
			if mode == 0: 
				_,Loss = self.Session.run([self.Optimizer, self.Cost], feed_dict={self.Inputs: Inputs, self.GroundTruths: GroundTruths})
				print("TRAINING || Epoch:{0}, Loss:{1:.2f}, Progress:{2:.2f} %".format(Epoch,Loss,Steps*100.0*self.InitParameters["BatchSize"]/len(self.TrainDictionary.keys())), end='\r')
				self.TrainLossTracker.append(Loss)
			if mode ==1:
				
				Outputs,Loss = self.Session.run([ self.Outputs, self.Cost], feed_dict={self.Inputs: Inputs, self.GroundTruths: GroundTruths})
				print(" TESTING || Loss:{0:.2f}, Progress:{1:.2f} %".format(Loss,Steps*100.0*self.InitParameters["BatchSize"]/len(self.TestDictionary.keys())), end='\r')
				self.UpdateConfusionMatrix(Outputs=Outputs,GroundTruths=GroundTruths)
				self.TestLossTracker.append(Loss)
				
	def Train(self):
		for Epoch in range(self.InitParameters["Epochs"]):
			self.ConfusionMatrix=np.zeros((self.InitParameters["NumClasses"],self.InitParameters["NumClasses"]))
			self.CounterMatrix=np.zeros(self.InitParameters["NumClasses"])
			self.RunEpoch(mode=1,ConfusionMatrix=self.ConfusionMatrix)
			self.GenerateMetrics(Epoch=Epoch)
			
			self.RunEpoch(mode=0,Epoch=Epoch)
			print("Model saved at {0}".format(self.Saver.save(self.Session,self.InitParameters["SavePath"]+"Epoch-"+str(Epoch)+".ckpt")))
			
			
			

obj=Linknet()
