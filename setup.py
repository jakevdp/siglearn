from distutils.core import setup

DESCRIPTION = "Noisy Machine Learning and Modeling"
LONG_DESCRIPTION = """
siglearn: Noisy Machine Learning and Modeling
=============================================

Siglearn is an experimental repository for providing a well-defined,
scikit-learn-style API for performing modeling and machine learning tasks
on noisy data.

Often in scientific detector data, we have some estimate of the error in
observed points. Unfortunately, most classic machine learning approaches
are not built with data errors in mind. Siglearn is an attempt to begin
collecting implementations of algorithms which do handle data errors.

For more information, visit http://github.com/jakevdp/siglearn/
"""
NAME = "siglearn"
AUTHOR = "Jake VanderPlas"
AUTHOR_EMAIL = "jakevdp@uw.edu"
MAINTAINER = "Jake VanderPlas"
MAINTAINER_EMAIL = "jakevdp@uw.edu"
URL = 'http://github.com/jakevdp/siglearn'
DOWNLOAD_URL = 'http://github.com/jakevdp/siglearn'
LICENSE = 'BSD 3-clause'

import siglearn
VERSION = siglearn.__version__

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['siglearn',
            ],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4'],
     )
