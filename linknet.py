from tensorflow.contrib import slim
import tensorflow as tf
import numpy as np

encoder_dict={"1":{"m":64,"n":64},"2":{"m":64,"n":128},"3":{"m":128,"n":256},"4":{"m":256,"n":512}}
decoder_dict={"1":{"n":64,"m":64},"2":{"n":64,"m":128},"3":{"n":128,"m":256},"4":{"n":256,"m":512}}


def encoder_level_2b(inputs,encoder_number,kernelsize=3):
	conv_1b=slim.conv2d(inputs, encoder_dict[encoder_number]["n"], kernelsize, stride=1, rate=1,padding='SAME')
	conv_1b=tf.layers.batch_normalization(conv_1b)
	conv_2b=slim.conv2d(conv_1b, encoder_dict[encoder_number]["n"], kernelsize, stride=1, rate=1,padding='SAME')
	conv_2b=tf.layers.batch_normalization(conv_2b)
	return conv_2b

def encoder_level_2a(inputs,encoder_number,kernelsize=3):
	conv_1a=slim.conv2d(inputs, encoder_dict[encoder_number]["n"], kernel_size=kernelsize, stride=2, rate=1,padding='SAME')
	conv_1a=tf.layers.batch_normalization(conv_1a)
	conv_2a=slim.conv2d(conv_1a, encoder_dict[encoder_number]["n"], kernel_size=kernelsize,stride=1, rate=1,padding='SAME')
	conv_2a=tf.layers.batch_normalization(conv_2a)
	return conv_2a

def encoder_level_1(inputs,encoder_number):
	conv1=encoder_level_2a(inputs=inputs,encoder_number=encoder_number)
	conv1=tf.layers.batch_normalization(conv1)# not needed
	input_downsampled=slim.conv2d(inputs, encoder_dict[encoder_number]["n"], kernel_size=3, stride=2, rate=1,padding='SAME')
	input_downsampled=tf.layers.batch_normalization(input_downsampled)
	conv1_combined=tf.concat([input_downsampled,conv1],axis=3)
	
	conv2=encoder_level_2b(conv1_combined,encoder_number=encoder_number)
	conv2_combined=tf.concat([conv1_combined,conv2],axis=3)
	return conv2_combined

def decoder_block(inputs,decoder_number,kernelsize=3):
	
	dconv_1=slim.conv2d(inputs, decoder_dict[decoder_number]["m"]//4, kernel_size=1, stride=1, rate=1,padding='SAME')
	dconv_1=tf.layers.batch_normalization(dconv_1)
	dconv_2=slim.conv2d_transpose(dconv_1,decoder_dict[decoder_number]["m"]//4,kernel_size=[kernelsize,kernelsize],stride=2,padding='SAME')
	dconv_2=tf.layers.batch_normalization(dconv_2)
	dconv_3=slim.conv2d(dconv_2,decoder_dict[decoder_number]["m"]//4,kernel_size=1,stride=1,rate=1,padding='SAME')
	dconv_3=tf.layers.batch_normalization(dconv_3)

	return dconv_3

def pre_encoder(inputs):
	pre_enc_1=slim.conv2d(inputs,encoder_dict["1"]["m"], kernel_size=7, stride=2, rate=1,padding='SAME')
	pre_enc_2=slim.conv2d(pre_enc_1,encoder_dict["1"]["m"], kernel_size=3, stride=1, rate=1,padding='SAME')
	return pre_enc_2
def post_decoder(inputs,classes):
	post_decoder_dict={"a":{"m":64,"n":32},"b":{"m":32,"n":32},"c":{"m":32,"n":classes}}
	post_decoder_1=slim.conv2d_transpose(inputs,post_decoder_dict["a"]["n"],kernel_size=3,stride=2,padding='SAME')
	post_decoder_2=slim.conv2d(post_decoder_1,post_decoder_dict["b"]["n"], kernel_size=3, stride=1, rate=1,padding='SAME')
	post_decoder_3=slim.conv2d_transpose(post_decoder_2,post_decoder_dict["c"]["n"],kernel_size=3,stride=1,padding='SAME')
	return post_decoder_3



def building_block(inputs,classes=4):
	input_1=pre_encoder(inputs)
	encoder_1=encoder_level_1(input_1,"1")
	encoder_2=encoder_level_1(encoder_1,"2")
	encoder_3=encoder_level_1(encoder_2,"3")
	encoder_4=encoder_level_1(encoder_3,"4")

	decoder_4=decoder_block(encoder_4,"4")
	decode_encode_concat4=tf.concat([encoder_3,decoder_4],axis=3)

	decoder_3=decoder_block(decode_encode_concat4,"3")
	decode_encode_concat3=tf.concat([encoder_2,decoder_3],axis=3)

	decoder_2=decoder_block(decode_encode_concat3,"2")
	decode_encode_concat2=tf.concat([encoder_1,decoder_2],axis=3)

	decoder_1=decoder_block(decode_encode_concat2,"1")

	output=post_decoder(decoder_1,classes)

	return output
