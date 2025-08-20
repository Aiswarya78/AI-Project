const mongoose = require('mongoose');

const expenseSchema = new mongoose.Schema({
  purchaseRequestId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'PurchaseRequest',
    required: true
  },
  employeeId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  title: {
    type: String,
    required: true,
    trim: true
  },
  description: {
    type: String,
    trim: true
  },
  amount: {
    type: Number,
    required: true,
    min: 0
  },
  category: {
    type: String,
    required: true
  },
  billImage: {
    type: String,
    required: true
  },
  expenseDate: {
    type: Date,
    required: true
  },
  reimbursementStatus: {
    type: String,
    enum: ['Pending', 'Processed', 'Paid'],
    default: 'Pending'
  },
  paymentMethod: {
    type: String,
    enum: ['Bank Transfer', 'Check', 'Cash', 'Other']
  },
  paymentDate: Date,
  notes: String,
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// Update timestamp on save
expenseSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

module.exports = mongoose.model('Expense', expenseSchema);


