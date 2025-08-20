const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const jwt = require('jsonwebtoken');
const ExpenseRequest = require('../models/ExpenseRequest');
const User = require('../models/User');
const mongoose = require('mongoose'); // Added missing import

const JWT_SECRET = 'your-secret-key-change-in-production';

// Multer configuration for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });

// Middleware to verify JWT token
const authenticateToken = (req, res, next) => {
  const token = req.header('x-auth-token');
  if (!token) {
    return res.status(401).json({ message: 'Access denied. No token provided.' });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    res.status(400).json({ message: 'Invalid token.' });
  }
};

// Create new expense request
router.post('/', authenticateToken, async (req, res) => {
  try {
    const {
      requestType,
      title,
      description,
      amount,
      currency,
      category,
      priority,
      dueDate
    } = req.body;

    const user = await User.findById(req.user.userId);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    const expenseRequest = new ExpenseRequest({
      employeeId: req.user.userId,
      employeeName: user.name,
      employeeEmail: user.email,
      department: user.department,
      requestType,
      title,
      description,
      amount,
      currency,
      category,
      priority,
      dueDate: dueDate ? new Date(dueDate) : null
    });

    await expenseRequest.save();

    res.status(201).json({
      message: 'Expense request created successfully',
      expenseRequest
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get all expense requests for a user
router.get('/user', authenticateToken, async (req, res) => {
  try {
    const expenseRequests = await ExpenseRequest.find({ employeeId: req.user.userId })
      .sort({ requestDate: -1 });

    res.json(expenseRequests);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get specific expense request
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const expenseRequest = await ExpenseRequest.findById(req.params.id);
    
    if (!expenseRequest) {
      return res.status(404).json({ message: 'Expense request not found' });
    }

    // Check if user owns this request or is admin
    if (expenseRequest.employeeId.toString() !== req.user.userId && req.user.role !== 'admin') {
      return res.status(403).json({ message: 'Access denied' });
    }

    res.json(expenseRequest);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Upload invoice image
router.post('/:id/invoice', authenticateToken, upload.single('invoice'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

    const expenseRequest = await ExpenseRequest.findById(req.params.id);
    if (!expenseRequest) {
      return res.status(404).json({ message: 'Expense request not found' });
    }

    // Check if user owns this request
    if (expenseRequest.employeeId.toString() !== req.user.userId) {
      return res.status(403).json({ message: 'Access denied' });
    }

    expenseRequest.invoiceImage = req.file.path;
    await expenseRequest.save();

    res.json({
      message: 'Invoice uploaded successfully',
      invoiceImage: req.file.path
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Upload bill image
router.post('/:id/bill', authenticateToken, upload.single('bill'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

    const expenseRequest = await ExpenseRequest.findById(req.params.id);
    if (!expenseRequest) {
      return res.status(404).json({ message: 'Expense request not found' });
    }

    // Check if user owns this request
    if (expenseRequest.employeeId.toString() !== req.user.userId) {
      return res.status(403).json({ message: 'Access denied' });
    }

    expenseRequest.billImage = req.file.path;
    await expenseRequest.save();

    res.json({
      message: 'Bill uploaded successfully',
      billImage: req.file.path
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Update expense request
router.put('/:id', authenticateToken, async (req, res) => {
  try {
    const expenseRequest = await ExpenseRequest.findById(req.params.id);
    if (!expenseRequest) {
      return res.status(404).json({ message: 'Expense request not found' });
    }

    // Check if user owns this request
    if (expenseRequest.employeeId.toString() !== req.user.userId) {
      return res.status(403).json({ message: 'Access denied' });
    }

    // Only allow updates if status is pending
    if (expenseRequest.status !== 'pending') {
      return res.status(400).json({ message: 'Cannot update approved/rejected requests' });
    }

    const updatedRequest = await ExpenseRequest.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true }
    );

    res.json({
      message: 'Expense request updated successfully',
      expenseRequest: updatedRequest
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Delete expense request
router.delete('/:id', authenticateToken, async (req, res) => {
  try {
    const expenseRequest = await ExpenseRequest.findById(req.params.id);
    if (!expenseRequest) {
      return res.status(404).json({ message: 'Expense request not found' });
    }

    // Check if user owns this request
    if (expenseRequest.employeeId.toString() !== req.user.userId) {
      return res.status(403).json({ message: 'Access denied' });
    }

    // Only allow deletion if status is pending
    if (expenseRequest.status !== 'pending') {
      return res.status(400).json({ message: 'Cannot delete approved/rejected requests' });
    }

    await ExpenseRequest.findByIdAndDelete(req.params.id);

    res.json({ message: 'Expense request deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get expense statistics for user
router.get('/user/stats', authenticateToken, async (req, res) => {
  try {
    const stats = await ExpenseRequest.aggregate([
      { $match: { employeeId: mongoose.Types.ObjectId(req.user.userId) } },
      {
        $group: {
          _id: '$status',
          count: { $sum: 1 },
          totalAmount: { $sum: '$amount' }
        }
      }
    ]);

    const totalRequests = await ExpenseRequest.countDocuments({ employeeId: req.user.userId });
    const totalAmount = await ExpenseRequest.aggregate([
      { $match: { employeeId: mongoose.Types.ObjectId(req.user.userId) } },
      { $group: { _id: null, total: { $sum: '$amount' } } }
    ]);

    res.json({
      stats,
      totalRequests,
      totalAmount: totalAmount[0]?.total || 0
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

module.exports = router;


