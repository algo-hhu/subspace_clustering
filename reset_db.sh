#!/usr/bin/env bash
set -e # Exit on error
docker compose down -v
docker compose up -d
sleep 4
prisma generate
prisma db push