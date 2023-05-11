    def strokeset_to_image(self, image_size=(1000, 200)):

        strokeset = self.format_strokeset()

        img = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        self.scale_factor = min(image_size[0] / self.width, image_size[1] / self.height)

        target_center = tuple(dim // 2 for dim in image_size)

        strokeset = [(np.array(stroke)) * self.scale_factor + target_center for stroke in strokeset]
        strokeset = [stroke.astype(np.int32) for stroke in strokeset]

        cv2.fillPoly(img, strokeset, color=255)

        return img