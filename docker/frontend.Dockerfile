FROM node:16-alpine

WORKDIR /app

# Copy package files first
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY public/ ./public/
COPY src/ ./src/

# Set environment variables
ENV NODE_ENV=development
ENV REACT_APP_API_URL=http://localhost:8000
ENV CI=true

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"] 