import tensorflow as tf


class SessionManager:
    sess = None

    @staticmethod
    def get_session():
        if SessionManager.sess is None:
            SessionManager.sess = tf.InteractiveSession()
            SessionManager.sess.run(tf.initialize_all_variables())
            SessionManager.sess.run(tf.initialize_local_variables())
        return SessionManager.sess
