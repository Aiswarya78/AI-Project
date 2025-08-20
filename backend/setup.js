const fs = require('fs');
const path = require('path');

console.log('🚀 Expense Tracker Setup Script');
console.log('================================\n');

// Check if uploads directory exists
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
  console.log('✅ Created uploads directory');
} else {
  console.log('✅ Uploads directory already exists');
}

// Check if .env file exists
const envFile = path.join(__dirname, '.env');
if (!fs.existsSync(envFile)) {
  const envContent = `PORT=5000
MONGODB_URI=mongodb+srv://mercy1112:mercy1112@cluster0.8x8j3ya.mongodb.net/ExpenseTracker?retryWrites=true&w=majority&appName=Cluster0
JWT_SECRET=your-secret-key-change-in-production
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-app-password`;
  
  fs.writeFileSync(envFile, envContent);
  console.log('✅ Created .env file with default values');
  console.log('⚠️  Please update the .env file with your actual values');
} else {
  console.log('✅ .env file already exists');
}

console.log('\n📋 Next Steps:');
console.log('1. Install MongoDB Community Server from: https://www.mongodb.com/try/download/community');
console.log('2. Start MongoDB service');
console.log('3. Update .env file with your email credentials (optional)');
console.log('4. Run: npm run dev (to start both frontend and backend)');
console.log('5. Or run: npm run server (backend only) and npm start (frontend only)');

console.log('\n🎯 Default Credentials:');
console.log('Admin: admin@expensetracker.com / admin123');
console.log('Employee: employee@expensetracker.com / employee123');

console.log('\n✨ Setup complete!');
