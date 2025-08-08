import os  
from django.core.asgi import get_asgi_application  
from channels.routing import ProtocolTypeRouter, URLRouter  
from channels.auth import AuthMiddlewareStack  
import langchain_stream.routing  
  
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Django_React_Langchain_Stream.settings')  

application = ProtocolTypeRouter({  
  "http": get_asgi_application(),  
  "websocket": AuthMiddlewareStack(  
        URLRouter(  
            langchain_stream.routing.websocket_urlpatterns  
        )  
    ),  
})