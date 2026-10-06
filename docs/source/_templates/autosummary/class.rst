{{ fullname | escape | underline}}
{% set non_init_methods = methods | reject("equalto", "__init__") | list %}
{% set excluded_attributes = ["call_super_init", "dump_patches", "training"] %}
.. currentmodule:: {{ module }}

.. add toctree option to make autodoc generate the pages

.. autoclass:: {{ objname }}

   {% block attributes %}
   {% if attributes | reject("in", excluded_attributes) | list %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: .
   {% for item in attributes %}
      {% if item not in excluded_attributes %}
      ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if non_init_methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   {% for item in methods %}
      {%- if not item.startswith('__') and not (item == 'forward' and objname in ['GRNModelSDE', 'BifurcatingSDE']) %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}