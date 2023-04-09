server {
    listen 80;
    server_name ${DOMAIN} www.${DOMAIN};

    location /.well-known/acme-challenge/ {
        root /vol/www/;
    }

    location /static/ {
        alias /vol/static/;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}