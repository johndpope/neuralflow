import tensorflow as tf


class SessionManager:
    sess = None

    @classmethod
    def get_session(cls):
        if cls.sess is None:
            print("Initializing Session...")
            cls.sess = tf.Session()
            #SessionManager.sess.run(tf.initialize_all_variables())
            cls.sess.run(tf.initialize_local_variables())
        return cls.sess


