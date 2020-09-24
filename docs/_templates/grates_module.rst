{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

{% block functions %}
{% if functions %}

{% for item in functions %}
.. automethod:: {{ fullname }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}

{% for item in classes %}
.. autoclass:: {{ fullname }}.{{ item }}
    :members:
{%- endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}

{% for item in exceptions %}
.. autoclass:: {{ fullname }}.{{ item }}
    :members:
{%- endfor %}
{% endif %}
{% endblock %}
