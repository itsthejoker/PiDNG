import struct

from addict import Dict

Type = Dict(
    {
        # TIFF Type Format': (Tag TYPE value, Size in bytes of one instance)
        "Invalid": (0, 0),  # Should not be used
        "Byte": (1, 1),  # 8-bit unsigned
        "Ascii": (2, 1),  # 7-bit ASCII code
        "Short": (3, 2),  # 16-bit unsigned
        "Long": (4, 4),  # 32-bit unsigned
        "Rational": (5, 8),  # 2 Longs, numerator:denominator
        "Sbyte": (6, 1),  # 8 bit signed integer
        "Undefined": (7, 1),  # 8 bit byte containing anything
        "Sshort": (8, 2),  # 16 bit signed
        "Slong": (9, 4),  # 32 bit signed
        "Srational": (10, 8),  # 2 Slongs, numerator:denominator
        "Float": (11, 4),  # 32bit float IEEE
        "Double": (12, 8),  # 64bit double IEEE
        "IFD": (13, 4),  # IFD (Same as Long)
    }
)

Tag = Dict(
    {
        "Invalid": (0, Type.Invalid),
        "NewSubfileType": (254, Type.Long),
        "ImageWidth": (256, Type.Long),
        "ImageLength": (257, Type.Long),
        "BitsPerSample": (258, Type.Short),
        "Compression": (259, Type.Short),
        "PhotometricInterpretation": (262, Type.Short),
        "FillOrder": (266, Type.Short),
        "ImageDescription": (270, Type.Ascii),
        "Make": (271, Type.Ascii),
        "Model": (272, Type.Ascii),
        "StripOffsets": (273, Type.Long),
        "Orientation": (274, Type.Short),
        "SamplesPerPixel": (277, Type.Short),
        "RowsPerStrip": (278, Type.Short),
        "StripByteCounts": (279, Type.Long),
        "XResolution": (282, Type.Rational),
        "YResolution": (283, Type.Rational),
        "PlanarConfiguration": (284, Type.Short),
        "ResolutionUnit": (296, Type.Short),
        "Software": (305, Type.Ascii),
        "DateTime": (306, Type.Ascii),
        "Artist": (315, Type.Ascii),
        "Predictor": (317, Type.Short),
        "TileWidth": (322, Type.Short),
        "TileLength": (323, Type.Short),
        "TileOffsets": (324, Type.Long),
        "TileByteCounts": (325, Type.Long),
        "SubIFD": (330, Type.IFD),
        "XMP_Metadata": (700, Type.Undefined),
        "CFARepeatPatternDim": (33421, Type.Short),
        "CFAPattern": (33422, Type.Byte),
        "Copyright": (33432, Type.Ascii),
        "ExposureTime": (33434, Type.Rational),
        "FNumber": (33437, Type.Rational),
        "EXIF_IFD": (34665, Type.IFD),
        "ExposureProgram": (34850, Type.Short),
        "PhotographicSensitivity": (34855, Type.Short),
        "SensitivityType": (34864, Type.Short),
        "ExifVersion": (36864, Type.Undefined),
        "DateTimeOriginal": (36867, Type.Ascii),
        "ShutterSpeedValue": (37377, Type.Srational),
        "ApertureValue": (37378, Type.Rational),
        "ExposureBiasValue": (37380, Type.Srational),
        "MaxApertureValue": (37381, Type.Rational),
        "SubjectDistance": (37382, Type.Rational),
        "MeteringMode": (37383, Type.Short),
        "Flash": (37385, Type.Short),
        "FocalLength": (37386, Type.Rational),
        "TIFF_EP_StandardID": (37398, Type.Byte),
        "SubsecTime": (37520, Type.Ascii),
        "SubsecTimeOriginal": (37521, Type.Ascii),
        "FocalPlaneXResolution": (41486, Type.Rational),
        "FocalPlaneYResolution": (41487, Type.Rational),
        "FocalPlaneResolutionUnit": (41488, Type.Short),
        "FocalLengthIn35mmFilm": (41989, Type.Short),
        "EXIFPhotoBodySerialNumber": (42033, Type.Ascii),
        "EXIFPhotoLensModel": (42036, Type.Ascii),
        "DNGVersion": (50706, Type.Byte),
        "DNGBackwardVersion": (50707, Type.Byte),
        "UniqueCameraModel": (50708, Type.Ascii),
        "CFAPlaneColor": (50710, Type.Byte),
        "CFALayout": (50711, Type.Short),
        "LinearizationTable": (50712, Type.Short),
        "BlackLevelRepeatDim": (50713, Type.Short),
        "BlackLevel": (50714, Type.Short),
        "WhiteLevel": (50717, Type.Short),
        "DefaultScale": (50718, Type.Rational),
        "DefaultCropOrigin": (50719, Type.Long),
        "DefaultCropSize": (50720, Type.Long),
        "ColorMatrix1": (50721, Type.Srational),
        "ColorMatrix2": (50722, Type.Srational),
        "CameraCalibration1": (50723, Type.Srational),
        "CameraCalibration2": (50724, Type.Srational),
        "AnalogBalance": (50727, Type.Rational),
        "AsShotNeutral": (50728, Type.Rational),
        "BaselineExposure": (50730, Type.Srational),
        "BaselineNoise": (50731, Type.Rational),
        "BaselineSharpness": (50732, Type.Rational),
        "BayerGreenSplit": (50733, Type.Long),
        "LinearResponseLimit": (50734, Type.Rational),
        "CameraSerialNumber": (50735, Type.Ascii),
        "AntiAliasStrength": (50738, Type.Rational),
        "ShadowScale": (50739, Type.Rational),
        "DNGPrivateData": (50740, Type.Byte),
        "MakerNoteSafety": (50741, Type.Short),
        "CalibrationIlluminant1": (50778, Type.Short),
        "CalibrationIlluminant2": (50779, Type.Short),
        "BestQualityScale": (50780, Type.Rational),
        "RawDataUniqueID": (50781, Type.Byte),
        "ActiveArea": (50829, Type.Long),
        "CameraCalibrationSignature": (50931, Type.Ascii),
        "ProfileCalibrationSignature": (50932, Type.Ascii),
        "NoiseReductionApplied": (50935, Type.Rational),
        "ProfileName": (50936, Type.Ascii),
        "ProfileHueSatMapDims": (50937, Type.Long),
        "ProfileHueSatMapData1": (50938, Type.Float),
        "ProfileHueSatMapData2": (50939, Type.Float),
        "ProfileToneCurve": (50940, Type.Float),
        "ProfileEmbedPolicy": (50941, Type.Long),
        "ForwardMatrix1": (50964, Type.Srational),
        "ForwardMatrix2": (50965, Type.Srational),
        "PreviewApplicationName": (50966, Type.Ascii),
        "PreviewApplicationVersion": (50967, Type.Ascii),
        "PreviewSettingsDigest": (50969, Type.Byte),
        "PreviewColorSpace": (50970, Type.Long),
        "PreviewDateTime": (50971, Type.Ascii),
        "NoiseProfile": (51041, Type.Double),
        "TimeCodes": (51043, Type.Byte),
        "FrameRate": (51044, Type.Srational),
        "OpcodeList1": (51008, Type.Undefined),
        "OpcodeList2": (51009, Type.Undefined),
        "ReelName": (51081, Type.Ascii),
        "BaselineExposureOffset": (
            51109,
            Type.Srational,
        ),  # 1.4 Spec says rational but mentions negative values?
        "DefaultBlackRender": (51110, Type.Long),
        "NewRawImageDigest": (51111, Type.Byte),
    }
)

ID_2_TAG = {id[0]: tag for tag, id in Tag.items()}


class DNGHeader(object):
    def __init__(self):
        self.IFDOffset = 8

    def raw(self):
        return struct.pack("<sI", "II\x2A\x00", self.IFDOffset)


class DNGTag(object):
    def __init__(self, tagType=Tag.Invalid, value=[]):
        self.Type = tagType
        self.TagId = tagType[0]
        self.DataType = tagType[1]
        self.DataCount = len(value)
        self.DataOffset = 0

        self.subIFD = None

        self.set_value(value)

        self.DataLength = len(self.Value)
        self.selfContained = True if self.DataLength <= 4 else False

    def __repr__(self):
        msg = "{}:{}".format(ID_2_TAG[self.TagId], self.Value)
        return msg

    def set_value(self, value):
        if self.DataType == Type.Byte:
            self.Value = struct.pack("<%sB" % len(value), *value)
        elif self.DataType == Type.Short:
            self.Value = struct.pack("<%sH" % len(value), *value)
        elif self.DataType == Type.Long:
            self.Value = struct.pack("<%sL" % len(value), *value)
        elif self.DataType == Type.Sbyte:
            self.Value = struct.pack("<%sb" % len(value), *value)
        elif self.DataType == Type.Undefined:
            self.Value = struct.pack("<%sB" % len(value), *value)
        elif self.DataType == Type.Sshort:
            self.Value = struct.pack("<%sh" % len(value), *value)
        elif self.DataType == Type.Slong:
            self.Value = struct.pack("<%sl" % len(value), *value)
        elif self.DataType == Type.Float:
            self.Value = struct.pack("<%sf" % len(value), *value)
        elif self.DataType == Type.Double:
            self.Value = struct.pack("<%sd" % len(value), *value)
        elif self.DataType == Type.Rational:
            self.Value = struct.pack(
                "<%sL" % (len(value) * 2),
                *[item for sublist in value for item in sublist]
            )
        elif self.DataType == Type.Srational:
            self.Value = struct.pack(
                "<%sl" % (len(value) * 2),
                *[item for sublist in value for item in sublist]
            )
        elif self.DataType == Type.Ascii:
            self.Value = struct.pack(
                "<%ssx0L" % len(value), bytearray(value.encode("ascii"))
            )
            self.DataCount += 1
        elif self.DataType == Type.IFD:
            self.Value = "\x00\x00\x00\x00"
            self.subIFD = value[0]
        self.Value += str.encode(
            "\x00" * (((len(self.Value) + 3) & 0xFFFFFFFC) - len(self.Value))
        )

    def set_buffer(self, buf, tagOffset, dataOffset):
        self.buf = buf
        self.TagOffset = tagOffset
        self.DataOffset = dataOffset
        if self.subIFD:
            self.subIFD.set_buffer(buf, self.DataOffset)

    def data_len(self):
        if self.subIFD:
            return self.subIFD.data_len()
        if self.selfContained:
            return 0
        else:
            return (len(self.Value) + 3) & 0xFFFFFFFC

    def write(self):
        if not self.buf:
            raise RuntimeError("Buffer not initialized!")

        if self.subIFD:
            self.subIFD.write()
            tagData = struct.pack(
                "<HHII", self.TagId, Type.Long[0], self.DataCount, self.DataOffset
            )
            struct.pack_into("<12s", self.buf, self.TagOffset, tagData)
        else:
            if self.selfContained:
                tagData = struct.pack(
                    "<HHI4s", self.TagId, self.DataType[0], self.DataCount, self.Value
                )
                struct.pack_into("<12s", self.buf, self.TagOffset, tagData)
            else:
                tagData = struct.pack(
                    "<HHII",
                    self.TagId,
                    self.DataType[0],
                    self.DataCount,
                    self.DataOffset,
                )
                struct.pack_into("<12s", self.buf, self.TagOffset, tagData)
                struct.pack_into(
                    "<%ds" % (self.DataLength), self.buf, self.DataOffset, self.Value
                )


class DNG_IFD(object):
    def __init__(self):
        self.tags = []
        self.NextIFDOffset = 0

    def __repr__(self):
        out = ""
        for tag in sorted(self.tags, key=lambda x: x.TagId):
            out += "{}\n".format(tag)
        return out

    def set_buffer(self, buf, offset):
        self.buf = buf
        self.offset = offset
        currentDataOffset = offset + 2 + len(self.tags) * 12 + 4
        currentTagOffset = offset + 2
        for tag in sorted(self.tags, key=lambda x: x.TagId):
            tag.set_buffer(buf, currentTagOffset, currentDataOffset)
            currentTagOffset += 12
            currentDataOffset += tag.data_len()
            # currentDataOffset = (currentDataOffset + 3) & 0xFFFFFFFC

    def data_len(self):
        totalLength = 2 + len(self.tags) * 12 + 4
        for tag in sorted(self.tags, key=lambda x: x.TagId):
            totalLength += tag.data_len()
        return (totalLength + 3) & 0xFFFFFFFC

    def write(self):
        if not self.buf:
            raise RuntimeError("buffer not initialized")

        struct.pack_into("<H", self.buf, self.offset, len(self.tags))

        for tag in sorted(self.tags, key=lambda x: x.TagId):
            tag.write()

        struct.pack_into(
            "<I", self.buf, self.offset + 2 + len(self.tags) * 12, self.NextIFDOffset
        )


class DNG(object):
    def __init__(self):
        self.IFDs = []
        self.ImageDataStrips = []
        self.StripOffsets = {}

    def set_buffer(self, buf):
        self.buf = buf

        currentOffset = 8

        for ifd in self.IFDs:
            ifd.set_buffer(buf, currentOffset)
            currentOffset += ifd.data_len()

    def data_len(self):
        totalLength = 8
        for ifd in self.IFDs:
            totalLength += (ifd.data_len() + 3) & 0xFFFFFFFC

        for i in range(len(self.ImageDataStrips)):
            self.StripOffsets[i] = totalLength
            strip = self.ImageDataStrips[i]
            totalLength += (len(strip) + 3) & 0xFFFFFFFC

        return (totalLength + 3) & 0xFFFFFFFC

    def write(self):
        struct.pack_into(
            "<ccbbI", self.buf, 0, b"I", b"I", 0x2A, 0x00, 8
        )  # assume the first IFD happens immediately after header

        for ifd in self.IFDs:
            ifd.write()

        for i in range(len(self.ImageDataStrips)):
            self.buf[
                self.StripOffsets[i] : self.StripOffsets[i]
                + len(self.ImageDataStrips[i])
            ] = self.ImageDataStrips[i]
