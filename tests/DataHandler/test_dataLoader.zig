const std = @import("std");
const zant = @import("zant");
const DataLoader = zant.data_handler.data_loader.DataLoader;
const fs = std.fs;
const pkgAllocator = zant.utils.allocator;

test "tests description" {
    std.debug.print("\n--- Running dataLoader tests\n", .{});
}

test "DataLoader xNext Test" {
    var features = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };

    const labels = [_]u8{ 1, 0 };

    var labelSlices: [2]f32 = undefined;
    labelSlices[0] = @as(f32, labels[0]);
    labelSlices[1] = @as(f32, labels[1]);

    var featureSlices: [2][]f32 = undefined;
    featureSlices[0] = &features[0];
    featureSlices[1] = &features[1];

    const labelSlice: []f32 = &labelSlices;

    var loader = DataLoader(f32, f32, f32, 1, 2){ // Cambiato u8 a f32 per le etichette
        .X = &featureSlices,
        .y = labelSlice,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    const x1 = loader.xNext() orelse unreachable;
    try std.testing.expectEqual(f32, @TypeOf(x1[0]));
    try std.testing.expectEqual(1.0, x1[0]);
    try std.testing.expectEqual(2.0, x1[1]);
    try std.testing.expectEqual(3.0, x1[2]);

    const x2 = loader.xNext() orelse unreachable;
    try std.testing.expectEqual(f32, @TypeOf(x2[0]));
    try std.testing.expectEqual(4.0, x2[0]);
    try std.testing.expectEqual(5.0, x2[1]);
    try std.testing.expectEqual(6.0, x2[2]);

    const x3 = loader.xNext();
    try std.testing.expectEqual(null, x3);

    const y1 = loader.yNext() orelse unreachable;
    try std.testing.expectEqual(f32, @TypeOf(y1)); // Cambiato da u8 a f32
    try std.testing.expectEqual(1.0, y1);

    const y2 = loader.yNext() orelse unreachable;
    try std.testing.expectEqual(f32, @TypeOf(y2)); // Cambiato da u8 a f32
    try std.testing.expectEqual(0.0, y2);

    const y3 = loader.yNext();
    try std.testing.expectEqual(null, y3);
}

test "splitCSVLine tests" {
    const allocator = pkgAllocator.allocator;

    const originalLine: []const u8 = "1,2,3,4,5\n";
    const csvLine = try allocator.alloc(u8, originalLine.len);
    defer allocator.free(csvLine);
    @memcpy(csvLine, originalLine);
    const loader = DataLoader(f64, f64, u8, 1, 2);
    const result = try loader.splitCSVLine(csvLine, &allocator);
    defer allocator.free(result);
    const expected_values = [_][]const u8{ "1", "2", "3", "4", "5" };
    for (result, 0..) |column, i| {
        try std.testing.expect(std.mem.eql(u8, column, expected_values[i]));
    }
    try std.testing.expectEqual(result.len, 5);

    const emptyColumnsLine: []const u8 = "1,,3,,5\n";
    const csvLineEmpty = try allocator.alloc(u8, emptyColumnsLine.len);
    defer allocator.free(csvLineEmpty);
    @memcpy(csvLineEmpty, emptyColumnsLine);
    const resultEmpty = try loader.splitCSVLine(csvLineEmpty, &allocator);
    defer allocator.free(resultEmpty);
    const expected_values_empty = [_][]const u8{ "1", "", "3", "", "5" };
    for (resultEmpty, 0..) |column, i| {
        try std.testing.expect(std.mem.eql(u8, column, expected_values_empty[i]));
    }
    try std.testing.expectEqual(resultEmpty.len, 5);

    const spacesLine: []const u8 = "a, b, c , d ,e\n";
    const csvLineSpaces = try allocator.alloc(u8, spacesLine.len);
    defer allocator.free(csvLineSpaces);
    @memcpy(csvLineSpaces, spacesLine);
    const resultSpaces = try loader.splitCSVLine(csvLineSpaces, &allocator);
    defer allocator.free(resultSpaces);
    const expected_values_spaces = [_][]const u8{ "a", " b", " c ", " d ", "e" };
    for (resultSpaces, 0..) |column, i| {
        try std.testing.expect(std.mem.eql(u8, column, expected_values_spaces[i]));
    }
    try std.testing.expectEqual(resultSpaces.len, 5);

    // Caso 4: Nessuna nuova riga finale
    const noNewLine: []const u8 = "1,2,3,4,5";
    const csvLineNoNewLine = try allocator.alloc(u8, noNewLine.len);
    defer allocator.free(csvLineNoNewLine);
    @memcpy(csvLineNoNewLine, noNewLine);
    const resultNoNewLine = try loader.splitCSVLine(csvLineNoNewLine, &allocator);
    defer allocator.free(resultNoNewLine);
    const expected_values_no_newline = [_][]const u8{ "1", "2", "3", "4", "5" };
    for (resultNoNewLine, 0..) |column, i| {
        try std.testing.expect(std.mem.eql(u8, column, expected_values_no_newline[i]));
    }
    try std.testing.expectEqual(resultNoNewLine.len, 5);

    // Caso 5: Delimitatori consecutivi (colonne vuote)
    const consecutiveDelimiters: []const u8 = ",,,,\n";
    const csvLineConsecutive = try allocator.alloc(u8, consecutiveDelimiters.len);
    defer allocator.free(csvLineConsecutive);
    @memcpy(csvLineConsecutive, consecutiveDelimiters);
    const resultConsecutive = try loader.splitCSVLine(csvLineConsecutive, &allocator);
    defer allocator.free(resultConsecutive);
    const expected_values_consecutive = [_][]const u8{ "", "", "", "", "" };
    for (resultConsecutive, 0..) |column, i| {
        try std.testing.expect(std.mem.eql(u8, column, expected_values_consecutive[i]));
    }
    try std.testing.expectEqual(resultConsecutive.len, 5);
}

test "readCSVLine test with correct file flags" {
    const allocator = pkgAllocator.allocator;
    const loader = DataLoader(f64, f64, u8, 1, 2);

    const cwd = std.fs.cwd();

    const temp_file_name = "test.csv";

    var file = try cwd.createFile(temp_file_name, .{
        .read = true,
        .truncate = true,
    });
    defer file.close();

    const csv_content: []const u8 = "1,2,3,4,5\n6,7,8,9,10\n";
    try file.writeAll(csv_content);

    try file.seekTo(0);

    var reader = file.reader();

    const lineBuf = try allocator.alloc(u8, 100);
    defer allocator.free(lineBuf);

    const firstLine = try loader.readCSVLine(&reader, lineBuf) orelse unreachable;
    try std.testing.expect(std.mem.eql(u8, firstLine, "1,2,3,4,5"));

    const secondLine = try loader.readCSVLine(&reader, lineBuf) orelse unreachable;
    try std.testing.expect(std.mem.eql(u8, secondLine, "6,7,8,9,10"));

    const eofLine = try loader.readCSVLine(&reader, lineBuf);
    try std.testing.expect(eofLine == null);
}

test "fromCSV test with feature and label extraction" {
    var allocator = pkgAllocator.allocator;
    var loader = DataLoader(f64, f64, u8, 1, 2){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    const file_name: []const u8 = "test.csv";
    const features = [_]usize{ 0, 1, 2, 3 };
    const featureCols: []const usize = &features;
    const labelCol: usize = 4;

    try loader.fromCSV(&allocator, file_name, featureCols, labelCol);

    try std.testing.expectEqual(loader.X.len, 2);
    try std.testing.expectEqual(loader.y.len, 2);

    try std.testing.expectEqual(loader.X[0][0], 1);
    try std.testing.expectEqual(loader.X[0][1], 2);
    try std.testing.expectEqual(loader.X[0][2], 3);
    try std.testing.expectEqual(loader.X[0][3], 4);

    try std.testing.expectEqual(loader.y[0], 5);

    try std.testing.expectEqual(loader.X[1][0], 6);
    try std.testing.expectEqual(loader.X[1][1], 7);
    try std.testing.expectEqual(loader.X[1][2], 8);
    try std.testing.expectEqual(loader.X[1][3], 9);

    try std.testing.expectEqual(loader.y[1], 10);

    loader.deinit(&allocator);
}

test "loadMNISTImages test" {
    var allocator = pkgAllocator.allocator;
    var loader = DataLoader(f64, f64, f64, 1, 2){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    const file_name: []const u8 = "datasets/t10k-images-idx3-ubyte";

    try loader.loadMNISTImages(&allocator, file_name);

    try std.testing.expectEqual(loader.X.len, 10000);

    // each image 28x28 pixels
    for (loader.X[0..10]) |image| {
        try std.testing.expectEqual(image.len, 28 * 28); // each image 784 pixels
    }

    loader.deinit(&allocator);
}

test "loadMNISTImages2D test Static" {
    var allocator = pkgAllocator.allocator;
    var loader = DataLoader(f64, f64, f64, 1, 3){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    const file_name: []const u8 = "datasets/t10k-images-idx3-ubyte";

    // Carica le immagini utilizzando la funzione aggiornata
    try loader.loadMNISTImages2DStatic(&allocator, file_name, 10000, 28, 28);

    // Verifica il numero totale di immagini
    try std.testing.expectEqual(loader.X.len, 10000);

    // Ogni immagine deve essere 28x28
    for (loader.X[0..10]) |image| {
        try std.testing.expectEqual(image.len, 28); // 28 righe
        for (image) |row| {
            try std.testing.expectEqual(row.len, 28); // 28 colonne per ogni riga
        }
    }

    // Deinizializza il caricatore
    loader.deinit(&allocator);
}

test "loadMNISTLabels test" {
    var allocator = pkgAllocator.allocator;
    var loader = DataLoader(f64, f64, f64, 1, 2){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    const file_name: []const u8 = "datasets/t10k-labels-idx1-ubyte";

    try loader.loadMNISTLabels(&allocator, file_name);

    try std.testing.expectEqual(loader.y.len, 10000);

    try std.testing.expectEqual(loader.y[0], 7);
    try std.testing.expectEqual(loader.y[1], 2);
    // try std.testing.expectEqual(loader.y[9999], 9);

    loader.deinit(&allocator);
}

test "loadMNISTDataParallel test" {
    var allocator = pkgAllocator.allocator;
    var loader = DataLoader(f64, f64, f64, 1, 2){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    const image_file_name: []const u8 = "datasets/t10k-images-idx3-ubyte";
    const label_file_name: []const u8 = "datasets/t10k-labels-idx1-ubyte";

    try loader.loadMNISTDataParallel(&allocator, image_file_name, label_file_name);

    try std.testing.expectEqual(loader.X.len, 10000);
    try std.testing.expectEqual(loader.y.len, 10000);

    loader.deinit(&allocator);
}
test "DataLoader shuffle simple test" {
    var features = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
        [_]f64{ 7.0, 8.0, 9.0 },
        [_]f64{ 10.0, 11.0, 12.0 },
    };

    const labels = [_]u8{ 1, 0, 1, 0 };

    // Effettua il cast esplicito da u8 a f64 per ogni elemento delle etichette
    var labelSlices: [4]f64 = undefined;
    labelSlices[0] = @as(f64, labels[0]);
    labelSlices[1] = @as(f64, labels[1]);
    labelSlices[2] = @as(f64, labels[2]);
    labelSlices[3] = @as(f64, labels[3]);

    var featureSlices: [4][]f64 = undefined;
    featureSlices[0] = &features[0];
    featureSlices[1] = &features[1];
    featureSlices[2] = &features[2];
    featureSlices[3] = &features[3];

    const labelSlice: []f64 = &labelSlices; // Modificato da `[]u8` a `[]f64` per coerenza con i tipi del DataLoader

    // Modifica del DataLoader per supportare `f64` come tipo per le etichette
    var loader = DataLoader(f64, f64, f64, 1, 2){ // Cambiato `u8` a `f64` per le etichette
        .X = &featureSlices,
        .y = labelSlice,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    var rng = std.Random.DefaultPrng.init(12345);

    const original_feature0 = loader.X[0];
    const original_label0 = loader.y[0];

    loader.shuffle(&rng);

    const new_feature0 = loader.X[0];

    const are_equal = std.mem.eql(f64, new_feature0, original_feature0);
    try std.testing.expect(!are_equal);

    var found_index: ?usize = null;
    for (loader.X, 0..) |feature, idx| {
        if (std.mem.eql(f64, feature, original_feature0)) {
            found_index = idx;
            break;
        }
    }
    try std.testing.expect(found_index != null);

    const corresponding_label = loader.y[found_index.?];
    try std.testing.expectEqual(original_label0, corresponding_label);
}

test "To Tensor Batch Test" {
    var features = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var allocator = pkgAllocator.allocator;

    const labels = [_]u8{ 1, 0 };

    var labelSlices: [2]f64 = undefined;
    labelSlices[0] = @as(f64, labels[0]);
    labelSlices[1] = @as(f64, labels[1]);

    var featureSlices: [2][]f64 = undefined;
    featureSlices[0] = &features[0];
    featureSlices[1] = &features[1];
    const labelSlice: []f64 = &labelSlices;

    var loader = DataLoader(f64, f64, u8, 1, 2){
        .X = &featureSlices,
        .y = labelSlice,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };
    var shapeXArr = [_]usize{ loader.batchSize, 3 };
    var shapeYArr = [_]usize{loader.batchSize};
    var shapeX: []usize = &shapeXArr;
    var shapeY: []usize = &shapeYArr;

    _ = loader.xNextBatch();
    _ = loader.yNextBatch();
    try loader.toTensor(&allocator, &shapeX, &shapeY);
    try std.testing.expect(loader.xTensor.shape[0] == 1);
    try std.testing.expect(loader.xTensor.shape[1] == 3);
    try std.testing.expect(loader.yTensor.shape[0] == 1);

    loader.xTensor.deinit();
    loader.yTensor.deinit();
}

test "MNIST batch and to Tensor test" {
    var allocator = pkgAllocator.allocator;
    var loader = DataLoader(f64, f64, f64, 2, 2){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };
    defer loader.deinit(&allocator);

    const image_file_name: []const u8 = "datasets/t10k-images-idx3-ubyte";
    const label_file_name: []const u8 = "datasets/t10k-labels-idx1-ubyte";

    try loader.loadMNISTDataParallel(&allocator, image_file_name, label_file_name);

    try std.testing.expectEqual(loader.X.len, 10000);
    try std.testing.expectEqual(loader.y.len, 10000);
    var shapeXArr = [_]usize{ loader.batchSize, 784 };
    var shapeYArr = [_]usize{loader.batchSize};
    var shapeX: []usize = &shapeXArr;
    var shapeY: []usize = &shapeYArr;
    _ = loader.xNextBatch();
    _ = loader.yNextBatch();
    try loader.toTensor(&allocator, &shapeX, &shapeY);
    try std.testing.expect(loader.xTensor.shape[0] == 2);
    try std.testing.expect(loader.xTensor.shape[1] == 784);
    try std.testing.expect(loader.yTensor.shape[0] == 2);

    loader.xTensor.deinit();
    loader.yTensor.deinit();
}
test "Shuffling and data split" {
    var allocator = pkgAllocator.allocator;
    var loader = DataLoader(f64, f64, f64, 32, 2){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };
    defer loader.deinit(&allocator);

    const image_file_name: []const u8 = "datasets/t10k-images-idx3-ubyte";
    const label_file_name: []const u8 = "datasets/t10k-labels-idx1-ubyte";
    var shapeXArr = [_]usize{ loader.batchSize, 784 };
    var shapeYArr = [_]usize{loader.batchSize};
    var shapeX: []usize = &shapeXArr;
    var shapeY: []usize = &shapeYArr;

    try loader.loadMNISTDataParallel(&allocator, image_file_name, label_file_name);
    try loader.trainTestSplit(&allocator, 0.2);
    const total_samples = loader.X.len;
    const train_samples = loader.X_train.?.len;
    const test_samples = loader.X_test.?.len;
    try std.testing.expect(test_samples == total_samples - train_samples);
    const x_batch = loader.xTrainNextBatch() orelse unreachable;
    const y_batch = loader.yTrainNextBatch() orelse unreachable;
    const x_testBatch = loader.xTestNextBatch() orelse unreachable;
    const y_testBatch = loader.yTestNextBatch() orelse unreachable;

    try std.testing.expect(x_batch.len == 32);
    try std.testing.expect(y_batch.len == 32);
    try std.testing.expect(x_testBatch.len == 32);
    try std.testing.expect(y_testBatch.len == 32);
    try loader.toTensor(&allocator, &shapeX, &shapeY);
    try std.testing.expect(loader.xTensor.shape[0] == 32);
    try std.testing.expect(loader.xTensor.shape[1] == 784);
    try std.testing.expect(loader.yTensor.shape[0] == 32);
    loader.xTensor.deinit();
    loader.yTensor.deinit();
}

test "Shuffling and data split 2D" {
    var allocator = pkgAllocator.allocator;
    var loader = DataLoader(f64, f64, f64, 32, 3){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };
    defer loader.deinit(&allocator);

    const image_file_name: []const u8 = "datasets/t10k-images-idx3-ubyte";
    const label_file_name: []const u8 = "datasets/t10k-labels-idx1-ubyte";

    var shapeXArr = [_]usize{ loader.batchSize, 28, 28 };
    var shapeYArr = [_]usize{loader.batchSize};
    var shapeX: []usize = &shapeXArr;
    var shapeY: []usize = &shapeYArr;

    try loader.loadMNIST2DDataParallel(&allocator, image_file_name, label_file_name);
    try loader.trainTestSplit(&allocator, 0.8);

    const total_samples = loader.X.len;
    const train_samples = loader.X_train.?.len;
    const test_samples = loader.X_test.?.len;

    try std.testing.expect(test_samples == total_samples - train_samples);

    const x_batch = loader.xTrainNextBatch() orelse unreachable;
    const y_batch = loader.yTrainNextBatch() orelse unreachable;
    const x_testBatch = loader.xTestNextBatch() orelse unreachable;
    const y_testBatch = loader.yTestNextBatch() orelse unreachable;

    try std.testing.expect(x_batch.len == 32);
    try std.testing.expect(y_batch.len == 32);
    try std.testing.expect(x_testBatch.len == 32);
    try std.testing.expect(y_testBatch.len == 32);

    try loader.toTensor(&allocator, &shapeX, &shapeY);

    try std.testing.expect(loader.xTensor.shape.len == 3);
    try std.testing.expect(loader.xTensor.shape[0] == 32);
    try std.testing.expect(loader.xTensor.shape[1] == 28);
    try std.testing.expect(loader.xTensor.shape[2] == 28);

    try std.testing.expect(loader.yTensor.shape.len == 1);
    try std.testing.expect(loader.yTensor.shape[0] == 32);

    loader.xTensor.deinit();
    loader.yTensor.deinit();
}
