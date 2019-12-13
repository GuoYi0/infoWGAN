# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import config as cfg
from noise_utils import create_infogan_noise_sample, create_gan_noise_sample
import tensorflow.layers as layers
from tf_utils import batch_norm
from os.path import join, realpath, dirname, basename, exists
from categorical_grid_plots import CategoricalPlotter
import progressbar
from random import shuffle
from tqdm import tqdm


SCRIPT_DIR = dirname(realpath(__file__))
PROJECT_DIR = SCRIPT_DIR


def load_mnist_dataset():
    mnist = input_data.read_data_sets("mnist", one_hot=True)
    pixel_height = 28
    pixel_width = 28
    n_channels = 1
    for dset in [mnist.train, mnist.validation, mnist.test]:
        num_images = len(dset.images)
        dset.images.shape = (num_images, pixel_height, pixel_width, n_channels)
    return mnist.train.images, mnist.train.labels


def constraint(v):
    if cfg.useWasserstein:
        return tf.clip_by_value(v, -cfg.clip_value, cfg.clip_value)
    else:
        return tf.identity(v)


def generatorNet(name, inputs, is_training, use_batchNorm, reuse=None):
    idx = 0
    f = inputs
    with tf.variable_scope(name, reuse=reuse):
        f = layers.dense(f, 1024, None, name="dense_%d" % idx)
        if use_batchNorm:
            f = layers.batch_normalization(f, training=is_training, name="bn_%d" % idx)
        f = tf.nn.relu(f, "relu_%d" % idx)

        idx += 1
        f = layers.dense(f, 7 * 7 * 128, None, name="dense_%d" % idx)  # 6272
        if use_batchNorm:
            f = layers.batch_normalization(f, training=is_training, name="bn_%d" % idx)
        f = tf.nn.relu(f, "relu_%d" % idx)

        f = tf.reshape(f, [-1, 7, 7, 128], name="reshape_%d" % idx)

        idx += 1
        f = layers.conv2d_transpose(f, 64, kernel_size=4, strides=2, padding="SAME", name="deconv_%d" % idx)
        if use_batchNorm:
            f = layers.batch_normalization(f, training=is_training, name="bn_%d" % idx)
        f = tf.nn.relu(f, "relu_%d" % idx)

        idx += 1
        f = layers.conv2d_transpose(f, 1, kernel_size=4, strides=2, padding="SAME", name="deconv_%d" % idx)
        f = tf.nn.sigmoid(f, "sigmoid_%d" % idx)

    return f


def discriminator(
        name, inputs, is_training, num_category, num_continuous, use_batchNorm=True, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        out = discriminatorNet(inputs, is_training, use_batchNorm)
        prob = layers.dense(out, 1, activation=None, name="discriminator_out", kernel_constraint=constraint)
        f = layers.dense(out, 128)
        f = tf.nn.leaky_relu(f, 0.01)
        f = layers.dense(f, units=num_category + num_continuous, activation=None)
    return {"prob_logits": prob, "q_logits": f}


def discriminatorNet(inputs, is_training, use_batchNorm):
    idx = 0
    f = inputs
    f = layers.conv2d(f, 64, kernel_size=4, strides=2, padding="SAME",
                      name="conv_%d" % idx, kernel_constraint=constraint)
    if use_batchNorm:
        f = batch_norm(f, is_training=is_training, channels=f.get_shape().as_list()[-1],
                       name="bn_%d" % idx, gradient_on_mean_varience=cfg.gradient_on_mean_varience)
        # f = layers.batch_normalization(f, training=is_training, name="bn_%d" % idx)
    f = tf.nn.leaky_relu(f, alpha=0.01, name="lrelu_%d" % idx)

    idx += 1
    f = layers.conv2d(f, 128, kernel_size=4, strides=2, padding="SAME",
                      name="conv_%d" % idx, kernel_constraint=constraint)
    if use_batchNorm:
        # f = layers.batch_normalization(f, training=is_training, name="bn_%d" % idx)
        f = batch_norm(f, is_training=is_training, channels=f.get_shape().as_list()[-1],
                       name="bn_%d" % idx, gradient_on_mean_varience=cfg.gradient_on_mean_varience)
    f = tf.nn.leaky_relu(f, alpha=0.01, name="lrelu_%d" % idx)

    idx += 1
    f = layers.flatten(f)
    f = layers.dense(f, 1024, name="dense_%d" % idx, kernel_constraint=constraint)
    f = tf.nn.leaky_relu(f, alpha=0.01, name="lrelu_%d" % idx)
    return f


def main(restore):
    X = load_mnist_dataset()
    dataset_name = "mnist"
    z_size = cfg.style_size + cfg.num_category + cfg.num_continuous
    sample_noise = create_infogan_noise_sample(cfg.num_category, cfg.num_continuous, cfg.style_size)
    # sample_noise(8),返回一个shape为(8, 74)的噪声数据，
    # 前十个为 one hot编码的均匀分布的0~9类别采用，然后是两个均匀分布的数，然后是62个正态分布的数
    discriminator_lr = tf.get_variable(
        "discriminator_lr", (),
        initializer=tf.constant_initializer(cfg.discriminator_lr), trainable=False)
    generator_lr = tf.get_variable(
        "generator_lr", (),
        initializer=tf.constant_initializer(cfg.generator_lr), trainable=False)
    n_images, image_height, image_width, n_channels = X[0].shape
    print("total images: ", n_images)
    true_images = tf.placeholder(tf.float32,
                                 [cfg.batch_size, image_height, image_width, n_channels], name="true_images")
    true_labels = tf.placeholder(tf.float32,
                                 [cfg.batch_size, cfg.num_category], name="true_labels")
    zc_vectors = tf.placeholder(tf.float32,
                                [None, z_size], name="zc_vectors")  # 输入进生成器的噪声
    is_training_discriminator = tf.placeholder(tf.bool, (), name="is_training_discriminator")
    is_training_generator = tf.placeholder(tf.bool, (), name="is_training_generator")
    fake_images = generatorNet(
        name="generator", inputs=zc_vectors, use_batchNorm=True, is_training=is_training_generator)
    disc_fake = discriminator(
        name="discriminator", inputs=fake_images, is_training=is_training_discriminator,
        use_batchNorm=cfg.use_BN_dis, reuse=None, num_category=cfg.num_category, num_continuous=cfg.num_continuous)
    disc_true = discriminator(
        name="discriminator", inputs=true_images, is_training=is_training_discriminator,
        use_batchNorm=cfg.use_BN_dis, reuse=True, num_category=cfg.num_category, num_continuous=cfg.num_continuous)

    # 训练判别器========================================================================================================
    logits_fake, logits_true = disc_fake["prob_logits"], disc_true["prob_logits"]
    if cfg.useWasserstein or cfg.useImprovWasserstein:
        discriminator_loss = tf.reduce_mean(logits_fake - logits_true)
    else:
        labels_bool = tf.concat([tf.zeros((cfg.batch_size,), tf.float32), tf.ones((cfg.batch_size,), tf.float32)],
                                axis=0)  # [假, 真]图片
        discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels_bool[..., tf.newaxis], logits=tf.concat([logits_fake, logits_true], axis=0)))
    dis_loss_summary = tf.summary.scalar("dis_loss", discriminator_loss)
    summary = [dis_loss_summary]
    if cfg.useInfoGan:
        labels_catrgory = tf.concat([zc_vectors[:, :cfg.num_category], true_labels], axis=0)  # [假，真] 类别标签
        q_cat_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=true_labels, logits=disc_true["q_logits"][:, :cfg.num_category]))
        q_cat_loss_summary = tf.summary.scalar("q_dis_cat_loss", q_cat_loss)
        discriminator_loss += cfg.categorical_lambda * q_cat_loss
        merge_dis_loss_summary = tf.summary.scalar("merge_dis_loss", discriminator_loss)
        summary.append(q_cat_loss_summary)
        summary.append(merge_dis_loss_summary)

        if cfg.num_continuous > 0:
            labels_continuous = zc_vectors[:, cfg.num_category: cfg.num_category + cfg.num_continuous]
            q_cont_loss = -0.5 * tf.reduce_mean(
                tf.square(labels_continuous - disc_true["q_logits"][:, cfg.num_category:]))
            discriminator_loss += cfg.continuous_lambda * q_cont_loss

    if cfg.useImprovWasserstein:
        alpha = tf.random_uniform(shape=[cfg.batch_size, 1, 1, 1], minval=0., maxval=1.)
        diff = fake_images - true_images
        interpolates = true_images + alpha * diff
        ys = discriminator(
            name="discriminator", inputs=interpolates, is_training=is_training_discriminator,
            use_batchNorm=cfg.use_BN_dis, reuse=True,
            num_category=cfg.num_category, num_continuous=cfg.num_continuous)
        ys = tf.squeeze(ys["prob_logits"], axis=-1)
        gradients = tf.gradients(ys, [interpolates], stop_gradients=[interpolates])[0]
        # ys: (256, );  interpolates: (256, 28, 28, 1);   gradients:(256, 28, 28, 1)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=(1, 2, 3)))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
        discriminator_loss += cfg.grad_lambda * gradient_penalty
        summary.append(tf.summary.scalar("grad_penalty", gradient_penalty))
    final_dis_loss = discriminator_loss
    if cfg.regularization:
        L2_loss = cfg.weight_decay * tf.add_n(
                [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables("discriminator")
                 if "bn" not in v.name])
        final_dis_loss += L2_loss
    discriminator_obj_summary = tf.summary.scalar("final_dis_loss", final_dis_loss)
    summary.append(discriminator_obj_summary)
    disc_summary = tf.summary.merge(summary)

    # 训练生成器 =======================================================================================================
    if cfg.useWasserstein:
        generator_loss = -tf.reduce_mean(logits_fake)
    else:
        labels_bool = tf.ones((cfg.batch_size,), tf.float32)
        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels_bool[..., tf.newaxis], logits=logits_fake))
    summary = [tf.summary.scalar("gen_loss", generator_loss)]
    if cfg.useInfoGan:
        labels_catrgory = zc_vectors[:, :cfg.num_category]
        q_gen_cat_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels_catrgory, logits=disc_fake["q_logits"][:, :cfg.num_category]))
        summary.append(tf.summary.scalar("q_gen_cat_loss", q_gen_cat_loss))
        generator_loss += cfg.categorical_lambda * q_gen_cat_loss
        summary.append(tf.summary.scalar("merge_gen_loss", generator_loss))
        if cfg.num_continuous > 0:
            labels_continuous = zc_vectors[:, cfg.num_category: cfg.num_category + cfg.num_continuous]
            q_cont_loss = -0.5 * tf.reduce_mean(tf.square(labels_continuous - disc_fake["q_logits"][:, cfg.num_category:]))
            generator_loss += cfg.continuous_lambda * q_cont_loss
    final_gen_loss = generator_loss
    if cfg.regularization:
        L2_loss = cfg.weight_decay * tf.add_n(
                [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables("generator")
                 if "bn" not in v.name])
        final_gen_loss += L2_loss
    summary.append(tf.summary.scalar("final_gen_loss", final_gen_loss))
    gen_summary = tf.summary.merge(summary)

    # discriminator_solver = tf.train.AdamOptimizer(learning_rate=discriminator_lr, beta1=0.5)
    # generator_solver = tf.train.AdamOptimizer(learning_rate=generator_lr, beta1=0.5)
    # discriminator_solver = tf.train.MomentumOptimizer(learning_rate=discriminator_lr, momentum=0.9)
    # generator_solver = tf.train.MomentumOptimizer(learning_rate=generator_lr, momentum=0.9)
    discriminator_solver = tf.train.RMSPropOptimizer(learning_rate=discriminator_lr)
    generator_solver = tf.train.RMSPropOptimizer(learning_rate=generator_lr)

    train_discriminator = discriminator_solver.minimize(final_dis_loss, var_list=tf.trainable_variables("discriminator"))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "discriminator")
    train_discriminator = tf.group(train_discriminator, update_ops)

    train_generator = generator_solver.minimize(final_gen_loss, var_list=tf.trainable_variables("generator"))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "generator")
    train_generator = tf.group(train_generator, update_ops)
    log_dir = next_unused_name(
        join(
            PROJECT_DIR,
            "%s_log" % (dataset_name,),
            "infogan" if cfg.useInfoGan else "gan"
        )
    )
    journalist = tf.summary.FileWriter(log_dir, flush_secs=10)
    plotter = CategoricalPlotter(
        journalist=journalist,
        categorical_cardinality=cfg.num_category,
        num_continuous=cfg.num_continuous,
        style_size=cfg.style_size,
        generate=lambda s, x: s.run(fake_images, {zc_vectors: x, is_training_discriminator: False,
                                                  is_training_generator: False}))
    idxes = np.arange(n_images, dtype=np.int32)
    iters = 0
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if restore:
            ckpt = tf.train.get_checkpoint_state("ckpt")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore from: ", ckpt.model_checkpoint_path)
        for epoch in range(1, cfg.n_epochs+1):
            disc_epoch_obj = []
            gen_epoch_obj = []
            if epoch % (cfg.n_epochs//2) == 0:
                sess.run(tf.assign(discriminator_lr, discriminator_lr.eval()*0.5))
                sess.run(tf.assign(generator_lr, generator_lr.eval()*0.5))
            shuffle(idxes)
            # pbar = create_progress_bar("epoch %d >> " % (epoch,))
            # 减去一个batchsize，以免最后不足一个
            for idx in tqdm(range(0, n_images-cfg.batch_size, cfg.batch_size)):
                for i in range(cfg.disc_iter):
                    batch = X[0][idxes[idx: idx + cfg.batch_size]]  # true image
                    labels = X[1][idxes[idx: idx + cfg.batch_size]]  # true image
                    noise = sample_noise(cfg.batch_size)
                    # 训练判别器
                    _, summary_result1, disc_obj = sess.run(
                        [train_discriminator, disc_summary, discriminator_loss],
                        feed_dict={
                            true_images: batch,
                            zc_vectors: noise,
                            is_training_discriminator: True,
                            is_training_generator: False,
                            true_labels: labels
                        }
                    )
                    disc_epoch_obj.append(disc_obj)

                # 训练生成器和互信息
                noise = sample_noise(cfg.batch_size)
                _, summary_result2, gen_obj = sess.run(
                    [train_generator,  gen_summary, generator_loss],
                    feed_dict={
                        zc_vectors: noise,
                        is_training_discriminator: False,
                        is_training_generator: True
                    }
                )
                journalist.add_summary(summary_result1, iters)
                journalist.add_summary(summary_result2, iters)
                journalist.flush()
                gen_epoch_obj.append(gen_obj)
                iters += 1

                if iters % cfg.plot_every == 0:
                    plotter.generate_images(sess, 10, iteration=iters)
                    journalist.flush()
                    ckpt_file = join(cfg.ckpt, "gan")
                    saver.save(sess, ckpt_file, iters)

            msg = "epoch %d >> discriminator loss %.2f (lr=%.6f), generator loss %.2f (lr=%.6f)" % (
                epoch,
                np.mean(disc_epoch_obj), sess.run(discriminator_lr),
                np.mean(gen_epoch_obj), sess.run(generator_lr)
            )
            print(msg)


def create_progress_bar(message):
    widgets = [
        message,
        progressbar.Counter(),
        ' ',
        progressbar.Percentage(),
        ' ',
        progressbar.Bar(),
        progressbar.AdaptiveETA()
    ]
    pbar = progressbar.ProgressBar(widgets=widgets)
    return pbar


def next_unused_name(name):
    save_name = name
    name_iteration = 0
    while exists(save_name):
        save_name = name + "-" + str(name_iteration)
        name_iteration += 1
    return save_name


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main(restore=False)
