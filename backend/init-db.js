const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const User = require('./models/User');
const Admin = require('./models/Admin');

const MONGODB_URI = 'mongodb+srv://mercy1112:mercy1112@cluster0.8x8j3ya.mongodb.net/ExpenseTracker?retryWrites=true&w=majority&appName=Cluster0';

async function initializeDatabase() {
  try {
    // Connect to MongoDB
    await mongoose.connect(MONGODB_URI);
    console.log('Connected to MongoDB');

    // Check if admin already exists
    const existingAdmin = await Admin.findOne({ email: 'admin@expensetracker.com' });
    if (!existingAdmin) {
      // Create default admin user
      const hashedPassword = await bcrypt.hash('admin123', 10);
      const admin = new Admin({
        name: 'System Administrator',
        email: 'admin@expensetracker.com',
        password: hashedPassword,
        role: 'admin',
        department: 'Finance'
      });
      await admin.save();
      console.log('Default admin user created: admin@expensetracker.com / admin123');
    } else {
      console.log('Admin user already exists');
    }

    // Check if sample user exists
    const existingUser = await User.findOne({ email: 'employee@expensetracker.com' });
    if (!existingUser) {
      // Create sample employee user
      const hashedPassword = await bcrypt.hash('employee123', 10);
      const user = new User({
        name: 'Sample Employee',
        email: 'employee@expensetracker.com',
        password: hashedPassword,
        employeeId: 'EMP001',
        department: 'Engineering',
        role: 'employee'
      });
      await user.save();
      console.log('Sample employee user created: employee@expensetracker.com / employee123');
    } else {
      console.log('Sample employee user already exists');
    }

    console.log('Database initialization completed successfully!');
    console.log('\nDefault Credentials:');
    console.log('Admin: admin@expensetracker.com / admin123');
    console.log('Employee: employee@expensetracker.com / employee123');
    console.log('\n⚠️  IMPORTANT: Change these passwords in production!');

  } catch (error) {
    console.error('Database initialization failed:', error);
  } finally {
    await mongoose.disconnect();
    console.log('Disconnected from MongoDB');
  }
}

// Run initialization if this file is executed directly
if (require.main === module) {
  initializeDatabase();
}

module.exports = initializeDatabase;


