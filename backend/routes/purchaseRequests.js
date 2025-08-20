const express = require('express');
const router = express.Router();
const jwt = require('jsonwebtoken');
const ExpenseRequest = require('../models/ExpenseRequest');
const User = require('../models/User');

const JWT_SECRET = 'your-secret-key-change-in-production';

// Middleware to verify User/Admin JWT token
const authenticateToken = (req, res, next) => {
  const token = req.header('x-auth-token');
  if (!token) {
    return res.status(401).json({ message: 'Access denied. No token provided.' });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded; // Contains userId/email/role per auth routes
    next();
  } catch (error) {
    res.status(400).json({ message: 'Invalid token.' });
  }
};

// Create a new purchase request with line items
router.post('/', authenticateToken, async (req, res) => {
  try {
    const {
      title,
      description,
      currency,
      category,
      priority,
      dueDate,
      items = []
    } = req.body;

    const user = await User.findById(req.user.userId);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    const purchaseRequest = new ExpenseRequest({
      employeeId: req.user.userId,
      employeeName: user.name,
      employeeEmail: user.email,
      department: user.department,
      requestType: 'purchase',
      title,
      description,
      currency: currency || 'USD',
      category,
      priority: priority || 'medium',
      dueDate: dueDate ? new Date(dueDate) : null,
      items: Array.isArray(items)
        ? items
            .filter((it) => it && it.description)
            .map((it) => ({
              description: String(it.description),
              quantity: Number(it.quantity || 1),
              unitPrice: Number(it.unitPrice || 0)
            }))
        : []
    });

    await purchaseRequest.save(); // model pre-save computes totals/amount

    res.status(201).json({
      message: 'Purchase request created successfully',
      purchaseRequest
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// List current user's purchase requests
router.get('/user', authenticateToken, async (req, res) => {
  try {
    const requests = await ExpenseRequest.find({
      employeeId: req.user.userId,
      requestType: 'purchase'
    }).sort({ requestDate: -1 });

    res.json(requests);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get specific purchase request
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const doc = await ExpenseRequest.findById(req.params.id);
    if (!doc || doc.requestType !== 'purchase') {
      return res.status(404).json({ message: 'Purchase request not found' });
    }

    if (doc.employeeId.toString() !== req.user.userId && req.user.role !== 'admin') {
      return res.status(403).json({ message: 'Access denied' });
    }

    res.json(doc);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Update a purchase request (only when pending and owned by user)
router.put('/:id', authenticateToken, async (req, res) => {
  try {
    const doc = await ExpenseRequest.findById(req.params.id);
    if (!doc || doc.requestType !== 'purchase') {
      return res.status(404).json({ message: 'Purchase request not found' });
    }

    if (doc.employeeId.toString() !== req.user.userId) {
      return res.status(403).json({ message: 'Access denied' });
    }

    if (doc.status !== 'pending') {
      return res.status(400).json({ message: 'Cannot update approved/rejected requests' });
    }

    const { title, description, currency, category, priority, dueDate, items } = req.body;

    if (title !== undefined) doc.title = title;
    if (description !== undefined) doc.description = description;
    if (currency !== undefined) doc.currency = currency;
    if (category !== undefined) doc.category = category;
    if (priority !== undefined) doc.priority = priority;
    if (dueDate !== undefined) doc.dueDate = dueDate ? new Date(dueDate) : null;

    if (Array.isArray(items)) {
      doc.items = items
        .filter((it) => it && it.description)
        .map((it) => ({
          description: String(it.description),
          quantity: Number(it.quantity || 1),
          unitPrice: Number(it.unitPrice || 0)
        }));
    }

    await doc.save(); // pre-save recalculates totals/amount

    res.json({ message: 'Purchase request updated successfully', purchaseRequest: doc });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Delete a purchase request (only when pending and owned by user)
router.delete('/:id', authenticateToken, async (req, res) => {
  try {
    const doc = await ExpenseRequest.findById(req.params.id);
    if (!doc || doc.requestType !== 'purchase') {
      return res.status(404).json({ message: 'Purchase request not found' });
    }

    if (doc.employeeId.toString() !== req.user.userId) {
      return res.status(403).json({ message: 'Access denied' });
    }

    if (doc.status !== 'pending') {
      return res.status(400).json({ message: 'Cannot delete approved/rejected requests' });
    }

    await ExpenseRequest.findByIdAndDelete(req.params.id);
    res.json({ message: 'Purchase request deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

module.exports = router;


