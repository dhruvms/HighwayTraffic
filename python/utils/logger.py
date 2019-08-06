# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
# Code referenced from https://becominghuman.ai/logging-in-tensorboard-with-pytorch-or-any-other-library-c549163dee9e
import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO         # Python 3.x
from PIL import Image

def get_model_name(args):
    model_name = ''

    if 0 in args.model_name:
        model_name = model_name + str(int(args.length)) + '-Length' + '_'
    if 1 in args.model_name:
        model_name = model_name + str(int(args.lanes)) + '-Lanes' + '_'
    if 2 in args.model_name:
        model_name = model_name + str(int(args.cars)) + '-Cars' + '_'
    if 3 in args.model_name:
        model_name = model_name + args.stadium*'Stadium' + (not args.stadium)*'Straight' + '_'
    if 4 in args.model_name:
        if args.both:
            model_name = model_name + 'Both_'
        else:
            if args.change:
                model_name = model_name + 'Change_'
            else:
                model_name = model_name + 'Follow_'
    if 5 in args.model_name:
        model_name = model_name + args.beta_dist*'Beta' + args.clamp_in_sim*'Clamped-Gaussian' + '_'
    if 6 in args.model_name:
        model_name = model_name + 'LR-' + str(args.lr) + '_'
    if 7 in args.model_name:
        model_name = model_name + 'Seed-' + str(args.seed) + '_'

    if model_name[-1] == '_':
        model_name = model_name[:-1]

    return model_name

class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.compat.v1.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def plot_summary(self, tag, figure, step):
        """Log a matplotlib figure."""

        s = BytesIO()
        figure.savefig(s, format='png')
        s.seek(0)
        img = Image.open(s)
        img_ar = np.array(img)

        img_summary = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img_ar.shape[0],
                                   width=img_ar.shape[1])

        summary = tf.compat.v1.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, step)
        self.writer.flush()
