import tensorflow as tf


class SessionManager:
    sess = None

    @staticmethod
    def get_session():
        if SessionManager.sess is None:
            SessionManager.sess = tf.InteractiveSession()
        return SessionManager.sess
