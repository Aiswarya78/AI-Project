module.exports = {
  // Server Configuration
  PORT: process.env.PORT || 5000,
  
  // MongoDB Configuration
  MONGODB_URI: process.env.MONGODB_URI || 'mongodb+srv://mercy1112:mercy1112@cluster0.8x8j3ya.mongodb.net/ExpenseTracker?retryWrites=true&w=majority&appName=Cluster0',
  
  // JWT Configuration
  JWT_SECRET: process.env.JWT_SECRET || 'your-secret-key-change-in-production',
  JWT_EXPIRES_IN: '24h',
  
  // Email Configuration
  EMAIL: {
    service: 'gmail',
    user: process.env.EMAIL_USER || 'your-email@gmail.com',
    pass: process.env.EMAIL_PASS || 'your-app-password'
  },
  
  // File Upload Configuration
  UPLOAD: {
    maxFileSize: 5 * 1024 * 1024, // 5MB
    allowedFileTypes: ['image/jpeg', 'image/png', 'image/gif', 'application/pdf'],
    uploadPath: 'uploads/'
  },
  
  // CORS Configuration
  CORS: {
    origin: process.env.FRONTEND_URL || 'http://localhost:3000',
    credentials: true
  },
  
  // Pagination
  PAGINATION: {
    defaultLimit: 10,
    maxLimit: 100
  },
  
  // Password Requirements
  PASSWORD: {
    minLength: 6,
    requireUppercase: false,
    requireLowercase: false,
    requireNumbers: false,
    requireSpecialChars: false
  }
};


