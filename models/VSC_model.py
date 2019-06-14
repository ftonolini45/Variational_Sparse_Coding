import numpy as np
import tensorflow as tf
import scipy.io as sio

from vae11 import vae_ss_ss_vamp_ept_sp
from vae11 import batch_manager
from vae11 import logging_utils
    
def train(x_data_train,params,save_dir):
        
    log = logging_utils.get_logger("lienar_projections")
    
    x_data = x_data_train    
    xsh = np.shape(x_data)
    z_dimension = params['z_dimension']
    sigma_s = 0
    
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    
    with graph.as_default():
    
        tf.set_random_seed(0)
        autoencoder = vae_ss_ss_vamp_ept_sp.VariationalAutoencoder("vae_autoencoder", xsh[1], z_dimension, params['h_enc_dim'], params['h_dec_dim'], params['n_pinputs'], sigma_s)
    
        x_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, xsh[1]], name="x_placeholder")
        alpha_ph = tf.placeholder(dtype=tf.float32, shape=(), name="alpha_ph")
        lam_ph = tf.placeholder(dtype=tf.float32, shape=(), name="lam_ph")
        z_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['z_dimension']], name="z_ph")
        nu_ph = tf.placeholder(dtype=tf.float32, shape=(), name="nu_ph")
        cost, reconstruction, reconstruction_t, opt_weights, z, zt, sd, cost_rec, cost_div, z_mean, z_log_sigma_sq, log_eta, msk, pseudo_inputs, zp, z_mean_p, z_log_sigma_sq_p, log_eta_p, sel = autoencoder.calculate_cost(x_placeholder, alpha_ph, lam_ph, z_ph, nu_ph)
        optimizer = tf.train.AdamOptimizer(params['initial_training_rate'])
        optimizer_slow = tf.train.AdamOptimizer(0.1*params['initial_training_rate'])
        minimize = optimizer.minimize(cost)
        minimize_slow = optimizer_slow.minimize(cost)
    
        mu_z_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['z_dimension']], name="mu_z_ph")
        sig_z_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['z_dimension']], name="sig_z_ph")
    
        z_samp,_ = autoencoder._sample_from_gaussian_dist(20, z_dimension, mu_z_ph, -0.3*tf.ones([20, z_dimension]), tf.log(1*tf.ones([20, z_dimension])),100)
    
        tf.summary.scalar("cost", cost)
        
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        session.run(init)
        train_writer = tf.summary.FileWriter('./train-ours/z20', graph=graph)
    log.debug("Initialised Graph")
        
    indices_generator = batch_manager.SequentialIndexer(params['batch_size'], xsh[0])

    alpha = params['alpha']
    wms = params['warm_up_start']
    wme = params['warm_up_end']
    wml = wme-wms
    lam = 0
    lamf = 1
    nu = params['nu']
#    nu_f = 100
#    dnu = nu-nu_f
    COST_PLOT = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])))
    LIK_PLOT = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])))
    KL_PLOT = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])))
    PIN_KL = np.zeros(np.int(np.round(params['num_iterations']/params['report_interval'])))
    ni = -1
    for i in range(params['num_iterations']):

        if i<wme and i>wms:
            lam = lam+lamf/wml;
    
        next_indices = indices_generator.next_indices()
        if i>params['shorten_steps_it']:
#            lam = 1;
            _, cost_value = session.run([minimize_slow, cost], feed_dict={x_placeholder: x_data[next_indices, :], alpha_ph: alpha, lam_ph: lam, z_ph: np.zeros([1,params['z_dimension']]), nu_ph: nu})
        else:
            _, cost_value = session.run([minimize, cost], feed_dict={x_placeholder: x_data[next_indices, :], alpha_ph: alpha, lam_ph: lam, z_ph: np.zeros([1,params['z_dimension']]), nu_ph: nu})    

        if i % params['report_interval'] == 0:
            ni = ni+1
            pin_KL, rec_lik, KL_div  = session.run([sd,cost_rec,cost_div], feed_dict={x_placeholder: x_data, alpha_ph: alpha, lam_ph: lam, z_ph: np.zeros([1,params['z_dimension']]), nu_ph: nu})
            COST_PLOT[ni] = -cost_value
            LIK_PLOT[ni] = -rec_lik
            KL_PLOT[ni] = KL_div
            PIN_KL[ni] = pin_KL
            print('--------------------------------------------------------------')
            print('Iteration Number ',i)
            print('Training ELBO:',-KL_div-rec_lik)
            print('Training Reconstruction Likelihood:',-rec_lik)
            print('Training KL Divergence:',KL_div)
            print('Training Aggregate Sparsity KL Divergence:',pin_KL)
            
            save_path = saver.save(session,save_dir)
            
        CF = {}
        CF['cost'] = COST_PLOT
        CF['KL_divergence'] = KL_PLOT
        CF['reconstruction_likelihood'] = LIK_PLOT
        CF['aggregate_KL'] = PIN_KL
        
    return CF

def encode(x_data_test,params,load_dir):
        
    log = logging_utils.get_logger("lienar_projections")
    
    x_data = x_data_test    
    xsh = np.shape(x_data)
    z_dimension = params['z_dimension']
    sigma_s = 0
    
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    
    with graph.as_default():
    
        tf.set_random_seed(0)
        autoencoder = vae_ss_ss_vamp_ept_sp.VariationalAutoencoder("vae_autoencoder", xsh[1], z_dimension, params['h_enc_dim'], params['h_dec_dim'], params['n_pinputs'], sigma_s)
    
        x_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, xsh[1]], name="x_placeholder")
        alpha_ph = tf.placeholder(dtype=tf.float32, shape=(), name="alpha_ph")
        lam_ph = tf.placeholder(dtype=tf.float32, shape=(), name="lam_ph")
        z_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['z_dimension']], name="z_ph")
        nu_ph = tf.placeholder(dtype=tf.float32, shape=(), name="nu_ph")
        cost, reconstruction, reconstruction_t, opt_weights, z, zt, sd, cost_rec, cost_div, z_mean, z_log_sigma_sq, log_eta, msk, pseudo_inputs, zp, z_mean_p, z_log_sigma_sq_p, log_eta_p, sel = autoencoder.calculate_cost(x_placeholder, alpha_ph, lam_ph, z_ph, nu_ph)
        optimizer = tf.train.AdamOptimizer(params['initial_training_rate'])
        optimizer_slow = tf.train.AdamOptimizer(0.1*params['initial_training_rate'])
        minimize = optimizer.minimize(cost)
        minimize_slow = optimizer_slow.minimize(cost)
    
        mu_z_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['z_dimension']], name="mu_z_ph")
        sig_z_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['z_dimension']], name="sig_z_ph")
    
        z_samp,_ = autoencoder._sample_from_gaussian_dist(20, z_dimension, mu_z_ph, -0.3*tf.ones([20, z_dimension]), tf.log(1*tf.ones([20, z_dimension])),100)
    
        tf.summary.scalar("cost", cost)
        
        init = tf.initialize_all_variables()
        session.run(init)
        saver = tf.train.Saver()
        saver.restore(session,load_dir)
        train_writer = tf.summary.FileWriter('./train-ours/z20', graph=graph)
    log.debug("Initialised Graph")

    alpha = params['alpha']
    lam = 1
    nu = params['nu']
    
    z_m = session.run(z, feed_dict={x_placeholder: x_data, alpha_ph: alpha, lam_ph: lam, z_ph: np.zeros([1,params['z_dimension']]), nu_ph:nu})
    log_z_act = session.run(log_eta, feed_dict={x_placeholder: x_data, alpha_ph: alpha, lam_ph: lam, z_ph: np.zeros([1,params['z_dimension']]), nu_ph:nu})
    z_act = np.exp(log_z_act)    
    
    return z_m, z_act

def decode(latent_codes,xsh,params,load_dir):
        
    log = logging_utils.get_logger("lienar_projections")
    
    z_dimension = params['z_dimension']
    sigma_s = 0
    
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    
    with graph.as_default():
    
        tf.set_random_seed(0)
        autoencoder = vae_ss_ss_vamp_ept_sp.VariationalAutoencoder("vae_autoencoder", xsh[1], z_dimension, params['h_enc_dim'], params['h_dec_dim'], params['n_pinputs'], sigma_s)
    
        z_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['z_dimension']], name="z_ph")
        x_mu, x_log_sigma_sq = autoencoder.calc_reconstruction(z_ph)
        x_sigma = tf.sqrt(tf.exp(x_log_sigma_sq))
#        cost, reconstruction, reconstruction_t, opt_weights, z, zt, sd, cost_rec, cost_div, z_mean, z_log_sigma_sq, log_eta, msk, pseudo_inputs, zp, z_mean_p, z_log_sigma_sq_p, log_eta_p, sel = autoencoder.calculate_cost(x_placeholder, alpha_ph, lam_ph, z_ph, nu_ph)
        
#        optimizer = tf.train.AdamOptimizer(params['initial_training_rate'])
#        optimizer_slow = tf.train.AdamOptimizer(0.1*params['initial_training_rate'])
#        minimize = optimizer.minimize(cost)
#        minimize_slow = optimizer_slow.minimize(cost)
    
#        mu_z_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['z_dimension']], name="mu_z_ph")
#        sig_z_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['z_dimension']], name="sig_z_ph")
#    
#        z_samp,_ = autoencoder._sample_from_gaussian_dist(20, z_dimension, mu_z_ph, -0.3*tf.ones([20, z_dimension]), tf.log(1*tf.ones([20, z_dimension])),100)
    
#        tf.summary.scalar("cost", cost)
        
        init = tf.initialize_all_variables()
        session.run(init)
        saver = tf.train.Saver()
        saver.restore(session,load_dir)
        train_writer = tf.summary.FileWriter('./train-ours/z20', graph=graph)
    log.debug("Initialised Graph")
    
    mu_x, sig_x = session.run([x_mu,x_sigma], feed_dict={z_ph: latent_codes})
    
    return mu_x, sig_x