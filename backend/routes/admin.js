const express = require('express');
const router = express.Router();
const jwt = require('jsonwebtoken');
const ExpenseRequest = require('../models/ExpenseRequest');
const User = require('../models/User');
const Admin = require('../models/Admin');

const JWT_SECRET = 'your-secret-key-change-in-production';

// Middleware to verify admin JWT token
const authenticateAdmin = (req, res, next) => {
  const token = req.header('x-auth-token');
  if (!token) {
    return res.status(401).json({ message: 'Access denied. No token provided.' });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    if (decoded.role !== 'admin') {
      return res.status(403).json({ message: 'Access denied. Admin only.' });
    }
    req.admin = decoded;
    next();
  } catch (error) {
    res.status(400).json({ message: 'Invalid token.' });
  }
};

// Get all expense requests (admin view)
router.get('/expenses', authenticateAdmin, async (req, res) => {
  try {
    const { status, department, page = 1, limit = 10 } = req.query;
    
    let query = {};
    
    if (status && status !== 'all') {
      query.status = status;
    }
    
    if (department && department !== 'all') {
      query.department = department;
    }

    const skip = (page - 1) * limit;
    
    const expenseRequests = await ExpenseRequest.find(query)
      .sort({ requestDate: -1 })
      .skip(skip)
      .limit(parseInt(limit))
      .populate('employeeId', 'name email employeeId department');

    const total = await ExpenseRequest.countDocuments(query);

    res.json({
      expenseRequests,
      totalPages: Math.ceil(total / limit),
      currentPage: parseInt(page),
      total
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get specific expense request details
router.get('/expenses/:id', authenticateAdmin, async (req, res) => {
  try {
    const expenseRequest = await ExpenseRequest.findById(req.params.id)
      .populate('employeeId', 'name email employeeId department phone');

    if (!expenseRequest) {
      return res.status(404).json({ message: 'Expense request not found' });
    }

    res.json(expenseRequest);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Update expense request status
router.put('/expenses/:id/status', authenticateAdmin, async (req, res) => {
  try {
    const { status, adminNotes } = req.body;
    
    if (!['pending', 'approved', 'rejected'].includes(status)) {
      return res.status(400).json({ message: 'Invalid status' });
    }

    const expenseRequest = await ExpenseRequest.findById(req.params.id);
    if (!expenseRequest) {
      return res.status(404).json({ message: 'Expense request not found' });
    }

    const admin = await Admin.findById(req.admin.adminId);
    
    expenseRequest.status = status;
    expenseRequest.adminNotes = adminNotes;
    expenseRequest.adminId = req.admin.adminId;
    expenseRequest.adminName = admin.name;
    expenseRequest.processedAt = new Date();

    await expenseRequest.save();

    // Send email notification to employee
    // This would be implemented with your email service

    res.json({
      message: 'Status updated successfully',
      expenseRequest
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get dashboard statistics
router.get('/dashboard/stats', authenticateAdmin, async (req, res) => {
  try {
    const totalRequests = await ExpenseRequest.countDocuments();
    const pendingRequests = await ExpenseRequest.countDocuments({ status: 'pending' });
    const approvedRequests = await ExpenseRequest.countDocuments({ status: 'approved' });
    const rejectedRequests = await ExpenseRequest.countDocuments({ status: 'rejected' });

    const totalAmount = await ExpenseRequest.aggregate([
      { $group: { _id: null, total: { $sum: '$amount' } } }
    ]);

    const approvedAmount = await ExpenseRequest.aggregate([
      { $match: { status: 'approved' } },
      { $group: { _id: null, total: { $sum: '$amount' } } }
    ]);

    const departmentStats = await ExpenseRequest.aggregate([
      {
        $group: {
          _id: '$department',
          count: { $sum: 1 },
          totalAmount: { $sum: '$amount' }
        }
      },
      { $sort: { count: -1 } }
    ]);

    const monthlyStats = await ExpenseRequest.aggregate([
      {
        $group: {
          _id: {
            year: { $year: '$requestDate' },
            month: { $month: '$requestDate' }
          },
          count: { $sum: 1 },
          totalAmount: { $sum: '$amount' }
        }
      },
      { $sort: { '_id.year': -1, '_id.month': -1 } },
      { $limit: 12 }
    ]);

    res.json({
      totalRequests,
      pendingRequests,
      approvedRequests,
      rejectedRequests,
      totalAmount: totalAmount[0]?.total || 0,
      approvedAmount: approvedAmount[0]?.total || 0,
      departmentStats,
      monthlyStats
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get all users
router.get('/users', authenticateAdmin, async (req, res) => {
  try {
    const users = await User.find({}, '-password').sort({ createdAt: -1 });
    res.json(users);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get user details
router.get('/users/:id', authenticateAdmin, async (req, res) => {
  try {
    const user = await User.findById(req.params.id, '-password');
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    const userExpenses = await ExpenseRequest.find({ employeeId: req.params.id })
      .sort({ requestDate: -1 });

    res.json({
      user,
      expenses: userExpenses
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Export expense data
router.get('/export', authenticateAdmin, async (req, res) => {
  try {
    const { startDate, endDate, status, department } = req.query;
    
    let query = {};
    
    if (startDate && endDate) {
      query.requestDate = {
        $gte: new Date(startDate),
        $lte: new Date(endDate)
      };
    }
    
    if (status && status !== 'all') {
      query.status = status;
    }
    
    if (department && department !== 'all') {
      query.department = department;
    }

    const expenses = await ExpenseRequest.find(query)
      .populate('employeeId', 'name email employeeId department')
      .sort({ requestDate: -1 });

    // Convert to CSV format (simplified)
    const csvData = expenses.map(expense => ({
      'Request ID': expense._id,
      'Employee Name': expense.employeeName,
      'Employee ID': expense.employeeId?.employeeId || '',
      'Department': expense.department,
      'Title': expense.title,
      'Description': expense.description,
      'Amount': expense.amount,
      'Currency': expense.currency,
      'Category': expense.category,
      'Priority': expense.priority,
      'Status': expense.status,
      'Request Date': expense.requestDate,
      'Processed Date': expense.processedAt || '',
      'Admin Notes': expense.adminNotes || ''
    }));

    res.json({
      message: 'Export successful',
      data: csvData,
      totalRecords: csvData.length
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Bulk status update
router.put('/expenses/bulk-status', authenticateAdmin, async (req, res) => {
  try {
    const { requestIds, status, adminNotes } = req.body;
    
    if (!['pending', 'approved', 'rejected'].includes(status)) {
      return res.status(400).json({ message: 'Invalid status' });
    }

    if (!Array.isArray(requestIds) || requestIds.length === 0) {
      return res.status(400).json({ message: 'No request IDs provided' });
    }

    const admin = await Admin.findById(req.admin.adminId);
    
    const updatePromises = requestIds.map(id => 
      ExpenseRequest.findByIdAndUpdate(id, {
        status,
        adminNotes,
        adminId: req.admin.adminId,
        adminName: admin.name,
        processedAt: new Date()
      })
    );

    await Promise.all(updatePromises);

    res.json({
      message: `Successfully updated ${requestIds.length} requests to ${status}`
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

module.exports = router;


