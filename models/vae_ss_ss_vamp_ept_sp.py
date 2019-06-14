import collections

import tensorflow as tf
import numpy as np
import math as m

from models import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

SMALL_CONSTANT = 1e-6

class VariationalAutoencoder(object):

    def __init__(self, name, n_input, n_hidden, n_layer_encoder, n_layer_decoder, n_pinputs, sigma_s, middle="gaussian"):
        
        self.sigma_s = sigma_s
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layer_encoder = n_layer_encoder
        self.n_layer_decoder = n_layer_decoder
        self.n_pinputs = n_pinputs
        self.name = name
        self.middle = middle
        self.bias_start = 0.0

        network_weights = self._create_weights()
        self.weights = network_weights

        self.nonlinearity = tf.nn.relu


    def calculate_cost(self, placeholder_in, alpha_ph,lam_ph,z_ph,nu_ph):
        x = placeholder_in
        alpha = alpha_ph
        lam = lam_ph
        zt = z_ph
        nu = nu_ph
        pseudo_inputs = self.weights['prior_param']['pseudoinputs']
        z_mean, z_log_sigma_sq, log_eta = self._calc_z_mean_and_sigma(x)
        z_mean_p, z_log_sigma_sq_p, log_eta_p = self._calc_z_mean_and_sigma(pseudo_inputs)
        log_eta_p = tf.log(tf.nn.sigmoid(nu*(tf.exp(log_eta_p)-0.5)))
        
        zp, _ = self._sample_from_gaussian_dist(tf.shape(pseudo_inputs)[0], self.n_hidden, z_mean_p, z_log_sigma_sq_p, log_eta_p, 100)
#        z1, msk = self._sample_from_gaussian_dist(tf.shape(x)[0], self.n_hidden, z_mean, z_log_sigma_sq, log_eta, 100)
        z, msk = self._sample_from_gaussian_dist(tf.shape(x)[0], self.n_hidden, (1-lam)*tf.zeros(tf.shape(z_mean))+lam*z_mean, (1-lam)*tf.zeros(tf.shape(z_mean))+lam*z_log_sigma_sq, log_eta, 100)
#        z = (1-lam)*zsp+lam*z1
        zer = tf.abs(zt)>0.2
        zer = tf.cast(zer, tf.float32)
        reconstruction = self.calc_reconstruction(z)
        reconstruction_t = self.calc_reconstruction(zt)
        reconstruction_p = self.calc_reconstruction(zp)
#        x_mu_p,x_log_sig_sq_p = self.calc_reconstruction(zp)
        x_mu_p = reconstruction_p[0]
        x_log_sig_sq_p = reconstruction_p[1]
        x_samp_p,_ = self._sample_from_gaussian_dist(tf.shape(pseudo_inputs)[0], tf.shape(pseudo_inputs)[1], x_mu_p, x_log_sig_sq_p, tf.zeros(tf.shape(pseudo_inputs)), 100)
        z_mean_ps, z_log_sigma_sq_ps, log_eta_ps = self._calc_z_mean_and_sigma(x_samp_p)
        sel = self._calc_selection(x)
#        sel = (1/self.n_pinputs)*tf.ones(tf.shape(sel))
        cost, sd, cost_rec, cost_div = self._calc_cost(x, alpha, lam, reconstruction, z_log_sigma_sq, z_mean, 100, log_eta, pseudo_inputs, reconstruction_p, z_mean_p, z_log_sigma_sq_p, log_eta_p, sel, z_mean_ps, z_log_sigma_sq_ps, log_eta_ps, zp)
        opt_weights = self.weights
        return cost, reconstruction, reconstruction_t, opt_weights, z, zt, sd, cost_rec, cost_div, z_mean, z_log_sigma_sq, log_eta, msk, pseudo_inputs, zp, z_mean_p, z_log_sigma_sq_p, log_eta_p, sel


    def calc_reconstruction(self, z):
        with tf.name_scope("decoder"):
            if self.middle == "bernoulli":
                hidden1_pre = tf.add(tf.matmul(z, self.weights['decoder']['W1_to_hidden']), self.weights['decoder']['b1_to_hidden'])
                hidden1_post = self.nonlinearity(hidden1_pre)
                
#                hidden3_pre = tf.add(tf.matmul(hidden1_post, self.weights['decoder']['W1c_htoh']), self.weights['decoder']['b1c_htoh'])
#                hidden3_post = self.nonlinearity(hidden3_pre)
#
#                hidden2_pre = tf.add(tf.matmul(hidden3_post, self.weights['decoder']['W1b_htoh']), self.weights['decoder']['b1b_htoh'])
#                hidden2_post = self.nonlinearity(hidden2_pre)

                y_pre = tf.add(tf.matmul(hidden1_post, self.weights['decoder']['W2_to_y_pre']), self.weights['decoder']['b2_to_y_pre'])
                y = tf.sigmoid(y_pre)
                return y
            elif self.middle == "gaussian":
                hidden1_pre = tf.add(tf.matmul(z,self.weights['decoder']['W3_to_hiddenG']), self.weights['decoder']['b3_to_hiddenG'])
                hidden1_post = self.nonlinearity(hidden1_pre)
#                hidden1_post = tf.nn.batch_normalization(hidden1_post,tf.Variable(tf.zeros([400], dtype=tf.float32)),tf.Variable(tf.ones([400], dtype=tf.float32)),None,None,0.000001,name="d_b_norm_1")
#                hidden1c_pre = tf.add(tf.matmul(hidden1_post, self.weights['decoder']['W3c_to_hiddenG']), self.weights['decoder']['b3c_to_hiddenG'])
#                hidden1c_post = self.nonlinearity(hidden1c_pre)
#                
#                hidden1d_pre = tf.add(tf.matmul(hidden1c_post, self.weights['decoder']['W3d_to_hiddenG']), self.weights['decoder']['b3d_to_hiddenG'])
#                hidden1d_post = self.nonlinearity(hidden1d_pre)
#                
#                hidden1e_pre = tf.add(tf.matmul(hidden1d_post, self.weights['decoder']['W3e_to_hiddenG']), self.weights['decoder']['b3e_to_hiddenG'])
#                hidden1e_post = self.nonlinearity(hidden1e_pre)
                
#                hidden1b_pre = tf.add(tf.matmul(hidden1_post, self.weights['decoder']['W3b_to_hiddenG']), self.weights['decoder']['b3b_to_hiddenG'])
#                hidden1b_post = self.nonlinearity(hidden1b_pre)
#                hidden1b_post = hidden1_post

                mu = tf.add(tf.matmul(hidden1_post, self.weights['decoder']['W4_to_muG']), self.weights['decoder']['b4_to_muG'])
                mu = tf.sigmoid(mu)  # see paper
                log_sigma_sq = tf.add(tf.matmul(hidden1_post, self.weights['decoder']['W5_to_log_sigmaG']), self.weights['decoder']['b5_to_log_sigmaG'])
                return mu, log_sigma_sq
            else:
                RuntimeError(self.middle + " is not yet constructed for reconstruction")


    def _calc_z_mean_and_sigma(self, x):
        with tf.name_scope("encoder"):
            hidden1_pre = tf.add(tf.matmul(x, self.weights['encoder']['W3_to_hidden']), self.weights['encoder']['b3_to_hidden'])
            hidden1_post = self.nonlinearity(hidden1_pre)
#            hidden1_post = tf.nn.batch_normalization(hidden1_post,tf.Variable(tf.zeros([400], dtype=tf.float32)),tf.Variable(tf.ones([400], dtype=tf.float32)),None,None,0.000001,name="e_b_norm_1")
#            hidden3_pre = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W3b_hth']), self.weights['encoder']['b3b_hth'])
#            hidden3_post = self.nonlinearity(hidden3_pre)
#            
#            hidden4_pre = tf.add(tf.matmul(hidden3_post, self.weights['encoder']['W3c_hth']), self.weights['encoder']['b3c_hth'])
#            hidden4_post = self.nonlinearity(hidden4_pre)
#            
#            hidden5_pre = tf.add(tf.matmul(hidden4_post, self.weights['encoder']['W3d_hth']), self.weights['encoder']['b3d_hth'])
#            hidden5_post = self.nonlinearity(hidden5_pre)
            
#            hidden2_pre = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W3_hth']), self.weights['encoder']['b3_hth'])
#            hidden2_post = self.nonlinearity(hidden2_pre)
#            hidden2_post = hidden1_post

            z_mean = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W4_to_mu']), self.weights['encoder']['b4_to_mu'])
            z_log_sigma_sq = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W5_to_log_sigma']), self.weights['encoder']['b5_to_log_sigma'])
            log_eta_pre = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W5_to_eta']), self.weights['encoder']['b5_to_eta']) 
            log_eta = -self.nonlinearity(-log_eta_pre)
#            log_eta = tf.log(tf.nn.sigmoid(100*(tf.exp(log_eta)-0.5)))
            
#            eta_pre = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W5_to_eta']), self.weights['encoder']['b5_to_eta']) 
#            eta_post = self.nonlinearity(eta_pre)
#            eta = 1-self.nonlinearity(1-eta_post)
#            log_eta = tf.log(eta)

#            log_eta_sq_pre = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W5_to_eta']), self.weights['encoder']['b5_to_eta']) 
#            log_sq_eta = -self.nonlinearity(-log_eta_sq_pre)
#            log_eta = tf.sqrt(log_sq_eta)
            
            tf.summary.histogram("z_mean", z_mean)
            tf.summary.histogram("z_log_sigma_sq", z_log_sigma_sq)
            tf.summary.histogram("log_eta", log_eta)
            return z_mean, z_log_sigma_sq, log_eta
        
    def _calc_selection(self, x):
        with tf.name_scope("encoder"):
            select1_pre = tf.add(tf.matmul(x, self.weights['prior_param']['selector_W1']), self.weights['prior_param']['selector_b1'])
            selection1 = tf.nn.sigmoid(select1_pre)
#            selection1 = tf.ones(tf.shape(selection1))
            
            siz_sel = tf.shape(selection1)
            v_norm = tf.norm(selection1,axis=1)
            V_norm = tf.expand_dims(v_norm,axis=1)
            S_norm = self.tf_repeat(V_norm, [1,siz_sel[1]])
            
            selection = tf.divide(selection1,S_norm)
#            selection = selection1
#            hidden1_post = tf.nn.batch_normalization(hidden1_post,tf.Variable(tf.zeros([400], dtype=tf.float32)),tf.Variable(tf.ones([400], dtype=tf.float32)),None,None,0.000001,name="e_b_norm_1")
#            hidden3_pre = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W3b_hth']), self.weights['encoder']['b3b_hth'])
#            hidden3_post = self.nonlinearity(hidden3_pre)
#            
#            hidden4_pre = tf.add(tf.matmul(hidden3_post, self.weights['encoder']['W3c_hth']), self.weights['encoder']['b3c_hth'])
#            hidden4_post = self.nonlinearity(hidden4_pre)
#            
#            hidden5_pre = tf.add(tf.matmul(hidden4_post, self.weights['encoder']['W3d_hth']), self.weights['encoder']['b3d_hth'])
#            hidden5_post = self.nonlinearity(hidden5_pre)
            
#            hidden2_pre = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W3_hth']), self.weights['encoder']['b3_hth'])
#            hidden2_post = self.nonlinearity(hidden2_pre)
#            hidden2_post = hidden1_post

#                z_mean = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W4_to_mu']), self.weights['encoder']['b4_to_mu'])
#                z_log_sigma_sq = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W5_to_log_sigma']), self.weights['encoder']['b5_to_log_sigma'])
#                log_eta_pre = tf.add(tf.matmul(hidden1_post, self.weights['encoder']['W5_to_eta']), self.weights['encoder']['b5_to_eta']) 
#                log_eta = -self.nonlinearity(-log_eta_pre)
#            log_eta = tf.log(tf.nn.sigmoid(100*(tf.exp(log_eta)-0.5)))
            
            return selection

    def _sample_from_gaussian_dist(self, num_rows, num_cols, mean, log_sigma_sq, log_eta,nu):
        with tf.name_scope("sample_in_z_space"):
            eps = tf.random_normal([num_rows, num_cols], 0, 1., dtype=tf.float32)
            nab = tf.random_uniform([num_rows, num_cols])
            msk = tf.nn.sigmoid(nu*(nab-(1-tf.exp(log_eta))))
            sample = tf.multiply(tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq)), eps)),msk)
        return sample, log_eta
    
    def tf_repeat(self, tensor, repeats):
        """
        Args:
    
        input: A Tensor. 1-D or higher.
        repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input
    
        Returns:
        
        A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
        """
        with tf.variable_scope("repeat"):
            expanded_tensor = tf.expand_dims(tensor, -1)
            multiples = [1] + repeats
            tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
            repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
        return repeated_tesnor

    def _calc_cost(self,x , alpha, lam, reconstruction, z_log_sigma_sq, z_mean, nu, log_eta, pseudo_inputs, reconstruction_p, z_mean_p, z_log_sigma_sq_p, log_eta_p, sel, z_mean_ps, z_log_sigma_sq_ps, log_eta_ps, zp):
        with tf.name_scope("cost"):

            with tf.name_scope("reconstruction_cost"):
                if self.middle == "bernoulli":
                    reconstr_loss_pre_sim_neg = x * tf.log(SMALL_CONSTANT + reconstruction) + (1-x) * tf.log(SMALL_CONSTANT + 1 - reconstruction)
                    reconstr_loss = - tf.reduce_sum(reconstr_loss_pre_sim_neg, 1)
                elif self.middle == "gaussian":
                    mu, log_sig_sq = reconstruction
                    mu_p, log_sig_sq_p = reconstruction_p

                    normalising_factor = - 0.5 * tf.log(SMALL_CONSTANT+tf.exp(log_sig_sq)) - 0.5 * np.log(2 * np.pi)
                    square_diff_between_mu_and_x = tf.square(mu - x)
                    inside_exp = -0.5 * tf.div(square_diff_between_mu_and_x,SMALL_CONSTANT+tf.exp(log_sig_sq))
                    
#                    normalising_factor_p = - 0.5 * tf.log(SMALL_CONSTANT+tf.exp(log_sig_sq_p)) - 0.5 * np.log(2 * np.pi)
#                    square_diff_between_mu_and_x_p = tf.square(mu_p - pseudo_inputs)
#                    inside_exp_p = -0.5 * tf.div(square_diff_between_mu_and_x_p,SMALL_CONSTANT+tf.exp(log_sig_sq_p))
                    
                    normalising_factor_p = - 0.5 * tf.log(SMALL_CONSTANT+tf.exp(z_log_sigma_sq_ps)) - 0.5 * np.log(2 * np.pi)
                    square_diff_between_mu_and_x_p = tf.square(z_mean_ps - zp)
                    inside_exp_p = -0.5 * tf.div(square_diff_between_mu_and_x_p,SMALL_CONSTANT+tf.exp(z_log_sigma_sq_ps))

                    tf.summary.scalar("log_sig_sq_min", tf.reduce_min(log_sig_sq))
                    tf.summary.scalar("log_sig_sq_mean", tf.reduce_mean(log_sig_sq))
                    tf.summary.scalar("log_sig_sq_max", tf.reduce_max(log_sig_sq))

                    tf.summary.scalar("normalising_factor_min", tf.reduce_min(normalising_factor))
                    tf.summary.scalar("square_diff_between_mu_and_x_min", tf.reduce_min(square_diff_between_mu_and_x))
                    tf.summary.scalar("square_diff_between_mu_and_x_max", tf.reduce_max(square_diff_between_mu_and_x))
                    tf.summary.scalar("square_diff_between_mu_and_x_mean", tf.reduce_mean(square_diff_between_mu_and_x))
                    tf.summary.scalar("inside_exp_min", tf.reduce_min(inside_exp))

                    reconstr_loss = -tf.reduce_sum(normalising_factor + inside_exp, 1)
                    reconstr_loss_p = -tf.reduce_sum(normalising_factor_p + inside_exp_p, 1)
                    reconstr_loss_p = tf.reduce_mean(reconstr_loss_p)
#                    reconstr_loss = tf.reduce_sum(tf.div(tf.square(mu - x),1),1)
                else:
                    RuntimeError(self.middle + " is not yet implemented as a cost")
                #reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.sub(reconstruction, x), 2.0))
                tf.summary.scalar("reconstr_loss", tf.reduce_mean(reconstr_loss))

            with tf.name_scope("latent_loss"):
                
                z_size = tf.shape(z_mean)
                
                sel = tf.expand_dims(sel,axis=2)
                sel = self.tf_repeat(sel, [1,1,z_size[1]])
                sel = tf.transpose(sel,perm=[0,2,1])
                
                zm = tf.expand_dims(z_mean,axis=2)
                zm = self.tf_repeat(zm, [1,1,self.n_pinputs])
                zlss = tf.expand_dims(z_log_sigma_sq,axis=2)
                zlss = self.tf_repeat(zlss, [1,1,self.n_pinputs])
                zle = tf.expand_dims(log_eta,axis=2)
                zle = self.tf_repeat(zle, [1,1,self.n_pinputs])
                zpm = tf.expand_dims(z_mean_p,axis=2)
                zpm = self.tf_repeat(zpm, [1,1,z_size[0]])
                zpm = tf.transpose(zpm,perm=[2,1,0])
                zplss = tf.expand_dims(z_log_sigma_sq_p,axis=2)
                zplss = self.tf_repeat(zplss, [1,1,z_size[0]])
                zplss = tf.transpose(zplss,perm=[2,1,0])
                zple = tf.expand_dims(log_eta_p,axis=2)
                zple = self.tf_repeat(zple, [1,1,z_size[0]])
                zple = tf.transpose(zple,perm=[2,1,0])
                
                # KL(q(z|x)||q(z|x_p)) KL(1||2)
                v_mean = zpm #2
#                v_mean = tf.zeros(tf.shape(zpm)) #2
                aux_mean = zm #1
                v_log_sig_sq = tf.log(tf.exp(zplss)+SMALL_CONSTANT) #2
#                v_log_sig_sq = tf.zeros(tf.shape(zpm)) #2
                aux_log_sig_sq = tf.log(tf.exp(zlss)+SMALL_CONSTANT) #1
                v_log_sig = tf.log(tf.sqrt(tf.exp(v_log_sig_sq))) #2
                aux_log_sig = tf.log(tf.sqrt(tf.exp(aux_log_sig_sq))) #1
                cost_KLN_a = v_log_sig-aux_log_sig+tf.divide(tf.exp(aux_log_sig_sq)+tf.square(aux_mean-v_mean),2*tf.exp(v_log_sig_sq))-0.5
                cost_KLN_b = tf.divide(tf.reduce_mean(tf.multiply(tf.multiply(tf.exp(zle),cost_KLN_a),sel),2),tf.reduce_mean(sel,2))
#                cost_KLN_b = tf.reduce_mean(tf.multiply(tf.exp(zle),cost_KLN_a),2)
                
                zpe = tf.exp(zple)
                cost_KLS_a = tf.multiply(tf.log(1-zpe+SMALL_CONSTANT)-tf.log(1-tf.exp(zle)+SMALL_CONSTANT),1-tf.exp(zle))+tf.multiply(zple-zle,tf.exp(zle))
                cost_KLS_b = -tf.divide(tf.reduce_mean(tf.multiply(cost_KLS_a,sel),2),tf.reduce_mean(sel,2))
#                cost_KLS_b = -tf.reduce_mean(cost_KLS_a,2)
                
                cost_KL = tf.reduce_sum(cost_KLN_b+cost_KLS_b,1)
                
                # KL(q(z|x_p)||p(z))
                
#                log_eta_p_mean = tf.log(tf.reduce_mean(tf.exp(log_eta_p),0))
#                log_eta_p_mean = tf.log(tf.divide(tf.reduce_mean(tf.multiply(tf.exp(zple),sel),2),tf.reduce_mean(sel,2)))
##                log_eta_p_mean = tf.log(tf.reduce_mean(tf.exp(zple),2))
#                latent_loss_spike2 = -tf.reduce_sum(tf.multiply(tf.log(1-alpha)-tf.log(1-tf.exp(log_eta_p_mean)+SMALL_CONSTANT),1-tf.exp(log_eta_p_mean))+tf.multiply(tf.log(alpha)-log_eta_p_mean,tf.exp(log_eta_p_mean)),1)
                
                # KL Average qp
                log_eta_p_mean = tf.log(tf.divide(tf.reduce_mean(tf.multiply(tf.exp(zple),sel),2),tf.reduce_mean(sel,2)))
                log_eta_p_mean = tf.log(tf.reduce_mean(tf.exp(log_eta_p_mean),1))
#                log_eta_p_mean = tf.log(tf.reduce_mean(tf.exp(zple),2))
                latent_loss_spike2 = -self.n_hidden*(tf.multiply(tf.log(1-alpha)-tf.log(1-tf.exp(log_eta_p_mean)+SMALL_CONSTANT),1-tf.exp(log_eta_p_mean))+tf.multiply(tf.log(alpha)-log_eta_p_mean,tf.exp(log_eta_p_mean)))
                
#                log_eta_p_mean = tf.log(tf.divide(tf.reduce_mean(tf.multiply(tf.exp(zple),sel),[2,1]),tf.reduce_mean(sel,[2,1])))
#                log_eta_p_mean = tf.expand_dims(log_eta_p_mean,axis=1)
#                log_eta_p_mean = self.tf_repeat(log_eta_p_mean, [1,z_size[1]])
#                log_eta_p_mean = tf.log(tf.divide(tf.reduce_mean(tf.multiply(tf.exp(zple),sel),2),tf.reduce_sum(sel,2)))
#                log_eta_p_mean = tf.log(tf.reduce_mean(tf.multiply(tf.exp(zple),sel),2))
#                log_eta_p_mean = tf.log(tf.reduce_mean(tf.multiply(tf.exp(zple),sel),2))
                
#                latent_loss_gaussian = -0.5*tf.reduce_sum(tf.multiply(1 + z_log_sigma_sq_p - tf.square(z_mean_p) - tf.exp(z_log_sigma_sq_p),tf.exp(log_eta_p)), 1)
#                latent_loss_spike1 = -tf.reduce_sum(tf.multiply(tf.log(1-alpha)-tf.log(1-tf.exp(log_eta_p_mean)+SMALL_CONSTANT),1-tf.exp(log_eta_p_mean))+tf.multiply(tf.log(alpha)-log_eta_p_mean,tf.exp(log_eta_p_mean)),1)
#                latent_loss_spike = -tf.reduce_sum(tf.multiply(tf.log(1-alpha)-tf.log(1-tf.exp(log_eta_p_mean)+SMALL_CONSTANT),1-tf.exp(log_eta_p_mean))+tf.multiply(tf.log(alpha)-log_eta_p_mean,tf.exp(log_eta_p_mean)))
                
#                latent_loss_gaussian = -0.5*tf.reduce_sum(tf.multiply(1 + z_log_sigma_sq_p - tf.square(z_mean_p) - tf.exp(z_log_sigma_sq_p)+2*tf.log(alpha)-2*log_eta_p,tf.exp(log_eta_p)), 1)
#                latent_loss_spike = -tf.reduce_sum(tf.multiply(tf.log(1-alpha)-tf.log(1-tf.exp(log_eta_p)+SMALL_CONSTANT),1-tf.exp(log_eta_p)), 1)
#                latent_loss = latent_loss_gaussian+latent_loss_spike
#                cost_KLP = tf.reduce_mean(latent_loss)
#                cost_KLP = tf.reduce_mean(latent_loss_gaussian)+latent_loss_spike
                
                cost_KLP = tf.reduce_mean(latent_loss_spike2)
#                cost_KLP = latent_loss_spike

                tf.summary.scalar("latent_loss", cost_KLP)

            cost = tf.reduce_mean(reconstr_loss + cost_KL) + cost_KLP
            cost_rec = tf.reduce_mean(reconstr_loss)
#            cost_rec = reconstr_loss_p
            cost_div = tf.reduce_mean(cost_KL)
            sd = cost_KLP
            
        return cost, sd, cost_rec, cost_div

    def _create_weights(self):
        all_weights = collections.OrderedDict()

        # Decoder
        all_weights['decoder'] = collections.OrderedDict()
        if self.middle == "gaussian":
            hidden_number_decoder = self.n_layer_decoder
            all_weights['decoder']['W3_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(self.n_hidden, hidden_number_decoder), dtype=tf.float32)
            all_weights['decoder']['b3_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
            
#            all_weights['decoder']['W3b_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
#            all_weights['decoder']['b3b_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
#            
#            all_weights['decoder']['W3c_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
#            all_weights['decoder']['b3c_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
##            
#            all_weights['decoder']['W3d_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
#            all_weights['decoder']['b3d_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)
#            
#            all_weights['decoder']['W3e_to_hiddenG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder), dtype=tf.float32)
#            all_weights['decoder']['b3e_to_hiddenG'] = tf.Variable(tf.zeros([hidden_number_decoder], dtype=tf.float32)  * self.bias_start)

            all_weights['decoder']['W4_to_muG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, self.n_input), dtype=tf.float32)
            all_weights['decoder']['b4_to_muG'] = tf.Variable(tf.zeros([self.n_input])  * self.bias_start, dtype=tf.float32)
            all_weights['decoder']['W5_to_log_sigmaG'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, self.n_input), dtype=tf.float32)
            all_weights['decoder']['b5_to_log_sigmaG'] = tf.Variable(tf.zeros([self.n_input])  * self.bias_start, dtype=tf.float32)
        elif self.middle == "bernoulli":
            hidden_number_decoder = self.n_layer_decoder
            all_weights['decoder']['W1_to_hidden'] = tf.Variable(vae_utils.xavier_init(self.n_hidden, hidden_number_decoder))
            tf.summary.histogram("W1_to_hidden", all_weights['decoder']['W1_to_hidden'])

            all_weights['decoder']['W1b_htoh'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder))
            tf.summary.histogram("W1b_htoh", all_weights['decoder']['W1_to_hidden'])
            
            all_weights['decoder']['W1c_htoh'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, hidden_number_decoder))
            tf.summary.histogram("W1c_htoh", all_weights['decoder']['W1_to_hidden'])

            all_weights['decoder']['W2_to_y_pre'] = tf.Variable(vae_utils.xavier_init(hidden_number_decoder, self.n_input))
            tf.summary.histogram("W2_to_y_pre", all_weights['decoder']['W1_to_hidden'])

            all_weights['decoder']['b1_to_hidden'] = tf.Variable(tf.ones([hidden_number_decoder], dtype=tf.float32) * self.bias_start)
            tf.summary.histogram("b1_to_hidden", all_weights['decoder']['b1_to_hidden'])

            all_weights['decoder']['b1b_htoh'] = tf.Variable(tf.ones([hidden_number_decoder], dtype=tf.float32) * self.bias_start)
            tf.summary.histogram("b1b_htoh", all_weights['decoder']['b1b_htoh'])
            
            all_weights['decoder']['b1c_htoh'] = tf.Variable(tf.ones([hidden_number_decoder], dtype=tf.float32) * self.bias_start)
            tf.summary.histogram("b1c_htoh", all_weights['decoder']['b1c_htoh'])

            all_weights['decoder']['b2_to_y_pre'] = tf.Variable(tf.ones([self.n_input], dtype=tf.float32) * self.bias_start)
            tf.summary.histogram("b2_to_y_pre", all_weights['decoder']['b2_to_y_pre'])

        else:
            raise RuntimeError

        # Encoder
        all_weights['encoder'] = collections.OrderedDict()
        hidden_number_encoder = self.n_layer_encoder
        all_weights['encoder']['W3_to_hidden'] = tf.Variable(vae_utils.xavier_init(self.n_input, hidden_number_encoder), dtype=tf.float32)
        tf.summary.histogram("W3_to_hidden", all_weights['encoder']['W3_to_hidden'])

#        all_weights['encoder']['W3_hth'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, hidden_number_encoder), dtype=tf.float32)
#        tf.summary.histogram("W3_hth", all_weights['encoder']['W3_hth'])
#        
#        all_weights['encoder']['W3b_hth'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, hidden_number_encoder), dtype=tf.float32)
#        tf.summary.histogram("W3b_hth", all_weights['encoder']['W3b_hth'])
#        
#        all_weights['encoder']['W3c_hth'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, hidden_number_encoder), dtype=tf.float32)
#        tf.summary.histogram("W3c_hth", all_weights['encoder']['W3c_hth'])
#        
#        all_weights['encoder']['W3d_hth'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, hidden_number_encoder), dtype=tf.float32)
#        tf.summary.histogram("W3d_hth", all_weights['encoder']['W3d_hth'])

        all_weights['encoder']['W4_to_mu'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, self.n_hidden),dtype=tf.float32)
        tf.summary.histogram("W4_to_mu", all_weights['encoder']['W4_to_mu'])
        
        all_weights['encoder']['W5_to_log_sigma'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, self.n_hidden), dtype=tf.float32)
        tf.summary.histogram("W5_to_log_sigma", all_weights['encoder']['W5_to_log_sigma'])
        
        all_weights['encoder']['W5_to_eta'] = tf.Variable(vae_utils.xavier_init(hidden_number_encoder, self.n_hidden), dtype=tf.float32)
        tf.summary.histogram("W5_to_eta", all_weights['encoder']['W5_to_log_sigma'])

        all_weights['encoder']['b3_to_hidden'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
        all_weights['encoder']['b3_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
        all_weights['encoder']['b3b_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
        all_weights['encoder']['b3c_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
        all_weights['encoder']['b3d_hth'] = tf.Variable(tf.zeros([hidden_number_encoder], dtype=tf.float32) * self.bias_start)
        all_weights['encoder']['b4_to_mu'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
        all_weights['encoder']['b5_to_log_sigma'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
        all_weights['encoder']['b5_to_eta'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
        
        all_weights['prior_param'] = collections.OrderedDict()
        all_weights['prior_param']['pseudoinputs'] = tf.Variable(vae_utils.xavier_init(self.n_pinputs,self.n_input), dtype=tf.float32)
        
        all_weights['prior_param']['selector_W1'] = tf.Variable(vae_utils.xavier_init(self.n_input,self.n_pinputs), dtype=tf.float32)
        all_weights['prior_param']['selector_b1'] = tf.Variable(tf.zeros([self.n_pinputs], dtype=tf.float32) * self.bias_start, dtype=tf.float32)
#        all_weights['prior_param']['alpha'] = tf.Variable(tf.ones(1,1), dtype=tf.float32)
        
        return all_weights