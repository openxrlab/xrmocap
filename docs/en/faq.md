# Frequently Asked Questions

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, do not hesitate to create an issue!

## Installation

- 'ImportError: libpng16.so.16: cannot open shared object file: No such file or directory'

  Please refer to [xrprimer faq](https://gitlab.bj.sensetime.com/openxrlab/xrprimer/-/blob/xrprimer_ee_dev/docs/en/faq.md).

- 'ImportError: liblapack.so.3: cannot open shared object file: No such file or directory'

  Please refer to [xrprimer faq](https://gitlab.bj.sensetime.com/openxrlab/xrprimer/-/blob/xrprimer_ee_dev/docs/en/faq.md).

- 'ModuleNotFoundError: No module named 'mmhuman3d.core.conventions.joints_mapping''

  Package `joints_mapping` actually exists in [github](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/joints_mapping), but it is not installed by pip for absence of `joints_mapping/__init__.py`. Install mmhuman3d from source will solve it:

  ```bash
  cd PATH_FOR_MMHUMAN3D
  git clone https://github.com/open-mmlab/mmhuman3d.git
  pip install -e ./mmhuman3d
  ```
