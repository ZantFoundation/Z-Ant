//! This module provides a data loader struct that can be used to load data from CSV files or the MNIST dataset.
//! in the future it will support other types of data.
//! it provides an iterator like interface to move in batch through the data.
//! It contains also the possibility to preprocess data and to shuffle the data.

const std = @import("std");
const zant = @import("../zant.zig");
const tensor = zant.core.tensor;

fn MagicalReturnType(comptime DataType: type, comptime dim_count: usize) type {
    return if (dim_count == 1) []DataType else []MagicalReturnType(DataType, dim_count - 1);
}
///It takes 3 comptime parameter end type and type of features and type of labels
pub fn DataLoader(comptime OutType: type, comptime Ftype: type, comptime LabelType: type, batchSize: i16, dimInput: usize) type {
    return struct {
        X: MagicalReturnType(OutType, dimInput),
        y: []OutType,
        x_index: usize = 0,
        y_index: usize = 0,
        xTensor: tensor.Tensor(OutType),
        yTensor: tensor.Tensor(OutType),
        batchSize: usize = batchSize,
        XBatch: MagicalReturnType(OutType, dimInput),
        yBatch: []OutType,

        XBuffer: ?[]OutType = null,

        X_train: ?MagicalReturnType(OutType, dimInput) = undefined,
        y_train: ?[]OutType = undefined,
        X_test: ?MagicalReturnType(OutType, dimInput) = undefined,
        y_test: ?[]OutType = undefined,
        x_train_index: usize = 0,
        y_train_index: usize = 0,
        x_test_index: usize = 0,
        y_test_index: usize = 0,

        ///X next  in an array like format
        pub fn xNext(self: *@This()) ?[]Ftype {
            const index = self.x_index;
            for (self.X[index..]) |x_row| {
                self.x_index += 1;
                return x_row;
            }
            return null;
        }

        ///Y next label iterator like
        pub fn yNext(self: *@This()) ?LabelType {
            const index = self.y_index;
            for (self.y[index..]) |label| {
                self.y_index += 1;
                return label;
            }
            return null;
        }

        ///Convert the data in the struct to a tensor
        pub fn toTensor(self: *@This(), allocator: *const std.mem.Allocator, shapeX: *[]usize, shapeY: *[]usize) !void {
            // Sconfeziona gli opzionali prima di passare i dati a fromArray
            const x_data = self.XBatch;
            const y_data = self.yBatch; // yBatch è non opzionale nel tuo esempio, se è opzionale fai lo stesso: self.yBatch.?

            self.xTensor = try tensor.Tensor(OutType).fromArray(allocator, x_data, shapeX.*);
            self.yTensor = try tensor.Tensor(OutType).fromArray(allocator, y_data, shapeY.*);
        }

        ///Reset the index of the iterator
        pub fn reset(self: *@This()) void {
            self.x_index = 0;
            self.y_index = 0;
            self.x_train_index = 0;
            self.y_train_index = 0;
            self.x_test_index = 0;
            self.y_test_index = 0;
        }
        //Maybe do batch size as a "attribute of the struct"
        ///Get the next batch of data
        pub fn xNextBatch(self: *@This()) ?[][]OutType {
            const start = self.x_index;
            const end = @min(start + self.batchSize, self.X.len);

            if (start >= end) return null;

            const batch = self.X[start..end];
            self.x_index = end;
            self.XBatch = batch;
            return batch;
        }
        ///Y next label iterator like in batch
        pub fn yNextBatch(self: *@This()) ?[]OutType {
            const start = self.y_index;
            const end = @min(start + self.batchSize, self.y.len);

            if (start >= end) return null;

            const batch = self.y[start..end];
            self.y_index = end;
            self.yBatch = batch;
            return batch;
        }
        pub fn trainTestSplit(self: *@This(), allocator: *const std.mem.Allocator, perc: f32) !void {
            if (perc <= 0.0 or perc >= 1.0) {
                return error.InvalidPercentage;
            }

            const total_samples = self.X.len;
            const total: f32 = @floatFromInt(total_samples);
            const train_size: usize = @intFromFloat(perc * total);

            var rng = std.rand.DefaultPrng.init(1234);
            self.shuffle(&rng);

            // Mantieni lo stesso livello di dimensioni
            self.X_train = try allocator.alloc(MagicalReturnType(OutType, dimInput - 1), train_size);
            self.y_train = try allocator.alloc(OutType, train_size);

            const test_size = total_samples - train_size;
            self.X_test = try allocator.alloc(MagicalReturnType(OutType, dimInput - 1), test_size);
            self.y_test = try allocator.alloc(OutType, test_size);

            const X_train = self.X_train.?;
            const y_train = self.y_train.?;
            const X_test = self.X_test.?;
            const y_test = self.y_test.?;

            for (self.X[0..train_size], 0..) |features, i| {
                X_train[i] = features;
            }
            for (self.y[0..train_size], 0..) |label, i| {
                y_train[i] = label;
            }

            for (self.X[train_size..], 0..) |features, i| {
                X_test[i] = features;
            }
            for (self.y[train_size..], 0..) |label, i| {
                y_test[i] = label;
            }
        }

        //We are using Knuth shuffle algorithm with complexity O(n)
        ///Shuffle the data using Knuth shuffle algorithm
        pub fn shuffle(self: *@This(), rng: *std.Random.DefaultPrng) void {
            const len = self.X.len;

            if (len <= 1) return;

            var i: usize = len - 1;
            while (true) {
                const j = rng.random().uintLessThan(usize, i + 1);

                const temp_feature = self.X[i];
                self.X[i] = self.X[j];
                self.X[j] = temp_feature;

                const temp_label = self.y[i];
                self.y[i] = self.y[j];
                self.y[j] = temp_label;

                if (i == 0) break;
                i -= 1;
            }
        }

        pub fn xTrainNextBatch(self: *@This()) ?MagicalReturnType(OutType, dimInput) {
            if (self.X_train == null) return null;
            const x_train = self.X_train.?;

            const start = self.x_train_index;
            const end = @min(start + self.batchSize, x_train.len);

            if (start >= end) return null;

            const batch = x_train[start..end];
            self.x_train_index = end;
            self.XBatch = batch;
            return batch;
        }

        pub fn yTrainNextBatch(self: *@This()) ?[]OutType {
            if (self.y_train == null) return null;
            const y_train = self.y_train.?;

            const start = self.y_train_index;
            const end = @min(start + self.batchSize, y_train.len);

            if (start >= end) return null;

            const batch = y_train[start..end];
            self.y_train_index = end;
            self.yBatch = batch;
            return batch;
        }

        pub fn xTestNextBatch(self: *@This()) ?MagicalReturnType(OutType, dimInput) {
            if (self.X_test == null) return null;
            const x_test = self.X_test.?;

            const start = self.x_test_index;
            const end = @min(start + self.batchSize, x_test.len);

            if (start >= end) return null;

            const batch = x_test[start..end];
            self.x_test_index = end;
            self.XBatch = batch;
            return batch;
        }

        pub fn yTestNextBatch(self: *@This()) ?[]OutType {
            if (self.y_test == null) return null;
            const y_test = self.y_test.?;

            const start = self.y_test_index;
            const end = @min(start + self.batchSize, y_test.len);

            if (start >= end) return null;

            const batch = y_test[start..end];
            self.y_test_index = end;
            self.yBatch = batch;
            return batch;
        }

        pub fn deinit(self: *@This(), allocator: *std.mem.Allocator) void {
            var features_freed = false;

            if (self.XBuffer) |buffer| {
                allocator.free(buffer);
                self.XBuffer = null;
            }

            if (self.X_train) |x_train| {
                for (x_train) |features| {
                    allocator.free(features);
                }
                allocator.free(x_train);
                features_freed = true;
            }

            if (self.y_train) |y_train| {
                allocator.free(y_train);
            }

            if (self.X_test) |x_test| {
                for (x_test) |features| {
                    allocator.free(features);
                }
                allocator.free(x_test);
                features_freed = true;
            }

            if (self.y_test) |y_test| {
                allocator.free(y_test);
            }

            if (!features_freed) {
                for (self.X) |features| {
                    allocator.free(features);
                }
                allocator.free(self.X);
                allocator.free(self.y);
            } else {
                allocator.free(self.X);
                allocator.free(self.y);
            }
        }
        ///Load data from a csv like file, it needs to take the file path and the columns of the features and the label
        pub fn fromCSV(self: *@This(), allocator: *const std.mem.Allocator, filePath: []const u8, featureCols: []const usize, labelCol: usize) !void {
            const file = try std.fs.cwd().openFile(filePath, .{});
            defer file.close();
            var reader = file.reader();
            const lineBuf = try allocator.alloc(u8, 1024);
            defer allocator.free(lineBuf);

            // Counts the numbers of rows in the CSV file really don't like this way of handling the file
            //since we are reading the file twice
            var numRows: usize = 0;
            while (true) {
                const maybeLine = try readCSVLine(&reader, lineBuf);
                if (maybeLine == null) break;
                numRows += 1;
            }
            // Reset the file to the beginning
            try file.seekTo(0);

            // Allocate memory for the features and labels
            self.X = try allocator.alloc([]OutType, numRows);
            self.y = try allocator.alloc(OutType, numRows);

            var rowIndex: usize = 0;
            while (true) {
                const maybeLine = try readCSVLine(&reader, lineBuf);
                if (maybeLine == null) break;

                const line = maybeLine.?;
                const columns = try splitCSVLine(line, allocator);
                defer freeCSVColumns(allocator, columns);

                // Allocate memory for the features of the current row
                self.X[rowIndex] = try allocator.alloc(Ftype, featureCols.len);

                // Parse the features
                for (featureCols, 0..) |colIndex, i| {
                    const valueStr = columns[colIndex];
                    const parsedIntValue = try parseXType(OutType, valueStr); // Parse the value as an integer first

                    if (@TypeOf(Ftype) == f32 or @TypeOf(Ftype) == f64) {
                        self.X[rowIndex][i] = @as(OutType, (parsedIntValue));
                    } else {
                        self.X[rowIndex][i] = @as(OutType, (parsedIntValue));
                    }
                }

                // Parse the label
                const labelValueStr = columns[labelCol];
                const parsedLabelIntValue = try parseYType(OutType, labelValueStr);

                if (@TypeOf(LabelType) == f32 or @TypeOf(LabelType) == f64) {
                    // Se LabelType è float, usa @floatFromInt per la conversione
                    self.y[rowIndex] = @as(OutType, @floatFromInt(parsedLabelIntValue));
                } else {
                    // Altrimenti, effettua il cast al tipo di output specificato
                    self.y[rowIndex] = @as(OutType, parsedLabelIntValue);
                }

                rowIndex += 1;
            }
        }
        ///Load data from the MNIST dataset, it needs to take the file path of the images and the labels
        pub fn loadMNISTImages(self: *@This(), allocator: *const std.mem.Allocator, filePath: []const u8) !void {
            const file = try std.fs.cwd().openFile(filePath, .{});
            defer file.close();
            var reader = file.reader();

            //  magic number (4 byte, big-endian)
            const magicNumber = try reader.readInt(u32, .big);
            if (magicNumber != 2051) {
                return error.InvalidFileFormat;
            }
            std.debug.print("Magic number: {d}\n", .{magicNumber});

            // num img (4 byte, big-endian)
            const numImages = try reader.readInt(u32, .big);

            // rows (4 byte, big-endian)
            const numRows = try reader.readInt(u32, .big);

            // columns (4 byte, big-endian)
            const numCols = try reader.readInt(u32, .big);

            // (28x28)
            if (numRows != 28 or numCols != 28) {
                return error.InvalidImageDimensions;
            }

            self.X = try allocator.alloc([]OutType, numImages);

            const imageSize = numRows * numCols;
            var i: usize = 0;

            while (i < numImages) {
                self.X[i] = try allocator.alloc(OutType, imageSize);

                const pixels = try allocator.alloc(u8, imageSize);
                defer allocator.free(pixels);

                try reader.readNoEof(pixels);

                var j: usize = 0;
                while (j < imageSize) {
                    self.X[i][j] = @as(OutType, @floatFromInt(pixels[j]));
                    j += 1;
                }

                i += 1;
            }
        }

        pub fn loadMNISTImages2D(
            self: *@This(),
            allocator: *const std.mem.Allocator,
            filePath: []const u8,
        ) !void {
            const file = try std.fs.cwd().openFile(filePath, .{});
            defer file.close();
            var reader = file.reader();

            // Magic number (4 byte, big-endian)
            const magicNumber = try reader.readInt(u32, .big);
            if (magicNumber != 2051) {
                return error.InvalidFileFormat;
            }
            std.debug.print("Magic number: {d}\n", .{magicNumber});

            // Number of images (4 byte, big-endian)
            const numImages = try reader.readInt(u32, .big);

            // Rows (4 byte, big-endian)
            const numRows = try reader.readInt(u32, .big);

            // Columns (4 byte, big-endian)
            const numCols = try reader.readInt(u32, .big);

            // Validate image dimensions
            if (numRows != 28 or numCols != 28) {
                return error.InvalidImageDimensions;
            }

            // Allocate space for all images
            self.X = try allocator.alloc([][]OutType, numImages);

            const imageSize = numRows * numCols;
            const imageBuffer = try allocator.alloc(u8, imageSize);
            defer allocator.free(imageBuffer);

            for (0..numImages) |i| {
                // Read the entire image (28x28) into the buffer
                try reader.readNoEof(imageBuffer);

                // Allocate space for the image (28 rows of 28 columns)
                self.X[i] = try allocator.alloc([]OutType, numRows);

                var rowStart: usize = 0;
                for (0..numRows) |row| {
                    self.X[i][row] = try allocator.alloc(OutType, numCols);

                    for (0..numCols) |col| {
                        self.X[i][row][col] = @as(OutType, @floatFromInt(imageBuffer[rowStart + col]));
                    }
                    rowStart += numCols;
                }
            }
        }

        pub fn loadMNISTImages2DStatic(
            self: *@This(),
            allocator: *const std.mem.Allocator,
            filePath: []const u8,
            numImages: usize,
            numRows: usize,
            numCols: usize,
        ) !void {
            const file = try std.fs.cwd().openFile(filePath, .{});
            defer file.close();
            var reader = file.reader();

            const magicNumber = try reader.readInt(u32, .big);
            if (magicNumber != 2051) {
                return error.InvalidFileFormat;
            }

            const datasetSize = numImages * numRows * numCols;

            // Alloca il buffer e memorizzalo in XBuffer
            self.XBuffer = try allocator.alloc(OutType, datasetSize);

            const pixelBuffer = try allocator.alloc(u8, datasetSize);
            defer allocator.free(pixelBuffer);
            try reader.readNoEof(pixelBuffer);

            for (0..datasetSize) |i| {
                self.XBuffer.?[i] = @as(OutType, @floatFromInt(pixelBuffer[i]));
            }

            self.X = try allocator.alloc(MagicalReturnType(OutType, dimInput - 1), numImages);

            for (0..numImages) |imageIdx| {
                self.X[imageIdx] = try allocator.alloc([]OutType, numRows);
                const imageOffset = imageIdx * numRows * numCols;
                for (0..numRows) |row| {
                    self.X[imageIdx][row] = self.XBuffer.?[imageOffset + row * numCols .. imageOffset + (row + 1) * numCols];
                }
            }
        }

        ///Load the labels from the MNIST dataset
        pub fn loadMNISTLabels(self: *@This(), allocator: *const std.mem.Allocator, filePath: []const u8) !void {
            const file = try std.fs.cwd().openFile(filePath, .{});
            defer file.close();
            var reader = file.reader();

            // Magic number (4 byte, big-endian)
            const magicNumber = try reader.readInt(u32, .big);
            if (magicNumber != 2049) {
                return error.InvalidFileFormat;
            }
            std.debug.print("Magic number (labels): {d}\n", .{magicNumber});

            // Number of labels (4 byte, big-endian)
            const numLabels = try reader.readInt(u32, .big);

            self.y = try allocator.alloc(OutType, numLabels);

            var i: usize = 0;
            while (i < numLabels) {
                const label = @as(OutType, @floatFromInt(try reader.readByte()));
                self.y[i] = label;
                i += 1;
            }
        }
        ///Load the data from the MNIST dataset in parallel
        pub fn loadMNISTDataParallel(self: *@This(), allocator: *const std.mem.Allocator, imageFilePath: []const u8, labelFilePath: []const u8) !void {
            const image_thread = try std.Thread.spawn(.{}, loadImages, .{ self, allocator, imageFilePath });
            defer image_thread.join();

            const label_thread = try std.Thread.spawn(.{}, loadLabels, .{ self, allocator, labelFilePath });
            defer label_thread.join();
        }

        pub fn loadMNIST2DDataParallel(self: *@This(), allocator: *const std.mem.Allocator, imageFilePath: []const u8, labelFilePath: []const u8) !void {
            const image_thread = try std.Thread.spawn(.{}, loadImages2D, .{ self, allocator, imageFilePath });
            defer image_thread.join();

            const label_thread = try std.Thread.spawn(.{}, loadLabels, .{ self, allocator, labelFilePath });
            defer label_thread.join();
        }

        fn loadImages(loader: *@This(), allocator: *const std.mem.Allocator, imageFilePath: []const u8) !void {
            try loader.loadMNISTImages(allocator, imageFilePath);
        }

        fn loadImages2D(loader: *@This(), allocator: *const std.mem.Allocator, imageFilePath: []const u8) !void {
            try loader.loadMNISTImages2DStatic(allocator, imageFilePath, 10000, 28, 28);
        }

        fn loadLabels(loader: *@This(), allocator: *const std.mem.Allocator, labelFilePath: []const u8) !void {
            try loader.loadMNISTLabels(allocator, labelFilePath);
        }
        ///Read a line from a CSV file
        pub fn readCSVLine(reader: *std.fs.File.Reader, lineBuf: []u8) !?[]u8 {
            const line = try reader.readUntilDelimiterOrEof(lineBuf, '\n');
            if (line) |l| {
                if (l.len == 0) return null;
                return l;
            }
            return null;
        }
        ///Split a CSV line
        pub fn splitCSVLine(line: []u8, allocator: *const std.mem.Allocator) ![]const []u8 {
            var columns = std.ArrayList([]u8).init(allocator.*);
            defer columns.deinit();

            var start: usize = 0;
            for (line, 0..) |c, i| {
                if (c == ',' or c == '\n') {
                    try columns.append(line[start..i]);
                    start = i + 1;
                }
            }
            if (start < line.len) {
                try columns.append(line[start..line.len]);
            }

            return columns.toOwnedSlice();
        }
        ///Free the columns of a CSV file
        fn freeCSVColumns(allocator: *const std.mem.Allocator, columns: []const []u8) void {
            allocator.free(columns);
        }
        ///Parse the X type
        fn parseXType(comptime XType: type, self: []const u8) !XType {
            const type_info = @typeInfo(XType);
            if (type_info == .Float) {
                return try std.fmt.parseFloat(XType, self);
            } else {
                return try std.fmt.parseInt(XType, self, 10);
            }
        }
        ///Parse the Y type
        fn parseYType(comptime YType: type, self: []const u8) !YType {
            const type_info = @typeInfo(YType);
            if (type_info == .Float) {
                return try std.fmt.parseFloat(YType, self);
            } else {
                return try std.fmt.parseInt(YType, self, 10);
            }
        }
    };
}
