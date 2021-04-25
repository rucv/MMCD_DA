from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.build_gen import *
from datasets.dataset_read import dataset_read


# Training settings
class Solver(object):
    def __init__(self, args, batch_size=64, source='svhn',
                 target='mnist', learning_rate=0.0002, interval=100, optimizer='adam'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10, num_c=2):
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        self.num_c = num_c
        if self.source == 'svhn' or self.target == 'svhn':
            self.scale = True
        else:
            self.scale = False
        print('dataset loading')
        self.datasets, self.dataset_test = dataset_read(source, target, self.batch_size, scale=self.scale,
                                                        all_use=self.all_use)
        print('load finished!')
        self.G = Generator(source=source, target=target)
        self.C1 = Classifier(source=source, target=target)
        self.C2 = Classifier(source=source, target=target)
        self.C3 = Classifier(source=source, target=target)
        self.C4 = Classifier(source=source, target=target)
        self.C5 = Classifier(source=source, target=target)
        self.C6 = Classifier(source=source, target=target)
        self.C7 = Classifier(source=source, target=target)

        if args.eval_only:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (
                    self.checkpoint_dir, self.source, self.target, self.checkpoint_dir, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))

        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.C3.cuda()
        self.C4.cuda()
        self.C5.cuda()
        self.C6.cuda()
        self.C7.cuda()

        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)

            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c3 = optim.SGD(self.C3.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c4 = optim.SGD(self.C4.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c5 = optim.SGD(self.C5.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c6 = optim.SGD(self.C6.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c7 = optim.SGD(self.C7.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c3 = optim.Adam(self.C3.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c4 = optim.Adam(self.C4.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c5 = optim.Adam(self.C5.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c6 = optim.Adam(self.C6.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c7 = optim.Adam(self.C7.parameters(),
                                     lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        self.opt_c3.zero_grad()
        self.opt_c4.zero_grad()
        self.opt_c5.zero_grad()
        self.opt_c6.zero_grad()
        self.opt_c7.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        self.C3.train()
        self.C4.train()
        self.C5.train()
        self.C6.train()
        self.C7.train()

        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            imgs = Variable(torch.cat((img_s, \
                                       img_t), 0))
            label_s = Variable(label_s.long().cuda())

            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()

            feat_s = self.G(img_s)

            if self.num_c == 3:
                #add c3---------------------------------------------------
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                output_s3 = self.C3(feat_s)
                #----------------------------------------------------------
            elif self.num_c == 4:
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                output_s3 = self.C3(feat_s)
                output_s4 = self.C4(feat_s)
                #----------------------------------------------------------
            elif self.num_c == 5:
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                output_s3 = self.C3(feat_s)
                output_s4 = self.C4(feat_s)
                output_s5 = self.C5(feat_s)
                #----------------------------------------------------------
            elif self.num_c == 6:
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                output_s3 = self.C3(feat_s)
                output_s4 = self.C4(feat_s)
                output_s5 = self.C5(feat_s)
                output_s6 = self.C6(feat_s)
                # ----------------------------------------------------------
            elif self.num_c == 7:
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                output_s3 = self.C3(feat_s)
                output_s4 = self.C4(feat_s)
                output_s5 = self.C5(feat_s)
                output_s6 = self.C6(feat_s)
                output_s7 = self.C7(feat_s)
                # ----------------------------------------------------------
            else:
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)


            if self.num_c == 3:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                loss_s3 = criterion(output_s3, label_s)
            elif self.num_c == 4:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                loss_s3 = criterion(output_s3, label_s)
                loss_s4 = criterion(output_s4, label_s)
            elif self.num_c == 5:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                loss_s3 = criterion(output_s3, label_s)
                loss_s4 = criterion(output_s4, label_s)
                loss_s5 = criterion(output_s5, label_s)
            elif self.num_c == 6:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                loss_s3 = criterion(output_s3, label_s)
                loss_s4 = criterion(output_s4, label_s)
                loss_s5 = criterion(output_s5, label_s)
                loss_s6 = criterion(output_s6, label_s)
            elif self.num_c == 7:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                loss_s3 = criterion(output_s3, label_s)
                loss_s4 = criterion(output_s4, label_s)
                loss_s5 = criterion(output_s5, label_s)
                loss_s6 = criterion(output_s6, label_s)
                loss_s7 = criterion(output_s7, label_s)
            else:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)

            if self.num_c == 3:
                loss_s = loss_s1 + loss_s2 + loss_s3
            elif self.num_c == 4:
                loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4
            elif self.num_c == 5:
                loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_s5
            elif self.num_c == 6:
                loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_s5 + loss_s6
            elif self.num_c == 7:
                loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_s5 + loss_s6 + loss_s7
            else:
                loss_s = loss_s1 + loss_s2

            loss_s.backward()
            self.opt_g.step()
            if self.num_c == 3:
                self.opt_c1.step()
                self.opt_c2.step()
                self.opt_c3.step()
            elif self.num_c == 4:
                self.opt_c1.step()
                self.opt_c2.step()
                self.opt_c3.step()
                self.opt_c4.step()
            elif self.num_c == 5:
                self.opt_c1.step()
                self.opt_c2.step()
                self.opt_c3.step()
                self.opt_c4.step()
                self.opt_c5.step()
            elif self.num_c == 6:
                self.opt_c1.step()
                self.opt_c2.step()
                self.opt_c3.step()
                self.opt_c4.step()
                self.opt_c5.step()
                self.opt_c6.step()
            elif self.num_c == 7:
                self.opt_c1.step()
                self.opt_c2.step()
                self.opt_c3.step()
                self.opt_c4.step()
                self.opt_c5.step()
                self.opt_c6.step()
                self.opt_c7.step()
            else:
                self.opt_c1.step()
                self.opt_c2.step()


            self.reset_grad()

            feat_s = self.G(img_s)
            if self.num_c == 3:
                #add c3---------------------------------------------------
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                output_s3 = self.C3(feat_s)
                #----------------------------------------------------------
            elif self.num_c == 4:
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                output_s3 = self.C3(feat_s)
                output_s4 = self.C4(feat_s)
                #----------------------------------------------------------
            elif self.num_c == 5:
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                output_s3 = self.C3(feat_s)
                output_s4 = self.C4(feat_s)
                output_s5 = self.C5(feat_s)
                #----------------------------------------------------------
            elif self.num_c == 6:
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                output_s3 = self.C3(feat_s)
                output_s4 = self.C4(feat_s)
                output_s5 = self.C5(feat_s)
                output_s6 = self.C6(feat_s)
                # ----------------------------------------------------------
            elif self.num_c == 7:
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                output_s3 = self.C3(feat_s)
                output_s4 = self.C4(feat_s)
                output_s5 = self.C5(feat_s)
                output_s6 = self.C6(feat_s)
                output_s7 = self.C7(feat_s)
                # ----------------------------------------------------------
            else:
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)

            feat_t = self.G(img_t)
            if self.num_c == 3:
                #add c3---------------------------------------------------
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                output_t3 = self.C3(feat_t)
                #----------------------------------------------------------
            elif self.num_c == 4:
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                output_t3 = self.C3(feat_t)
                output_t4 = self.C4(feat_t)
                #----------------------------------------------------------
            elif self.num_c == 5:
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                output_t3 = self.C3(feat_t)
                output_t4 = self.C4(feat_t)
                output_t5 = self.C5(feat_t)
                #----------------------------------------------------------
            elif self.num_c == 6:
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                output_t3 = self.C3(feat_t)
                output_t4 = self.C4(feat_t)
                output_t5 = self.C5(feat_t)
                output_t6 = self.C6(feat_t)
                # ----------------------------------------------------------
            elif self.num_c == 7:
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                output_t3 = self.C3(feat_t)
                output_t4 = self.C4(feat_t)
                output_t5 = self.C5(feat_t)
                output_t6 = self.C6(feat_t)
                output_t7 = self.C7(feat_t)
                # ----------------------------------------------------------
            else:
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)

            if self.num_c == 3:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                loss_s3 = criterion(output_s3, label_s)
            elif self.num_c == 4:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                loss_s3 = criterion(output_s3, label_s)
                loss_s4 = criterion(output_s4, label_s)
            elif self.num_c == 5:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                loss_s3 = criterion(output_s3, label_s)
                loss_s4 = criterion(output_s4, label_s)
                loss_s5 = criterion(output_s5, label_s)
            elif self.num_c == 6:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                loss_s3 = criterion(output_s3, label_s)
                loss_s4 = criterion(output_s4, label_s)
                loss_s5 = criterion(output_s5, label_s)
                loss_s6 = criterion(output_s6, label_s)
            elif self.num_c == 7:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)
                loss_s3 = criterion(output_s3, label_s)
                loss_s4 = criterion(output_s4, label_s)
                loss_s5 = criterion(output_s5, label_s)
                loss_s6 = criterion(output_s6, label_s)
                loss_s7 = criterion(output_s7, label_s)
            else:
                loss_s1 = criterion(output_s1, label_s)
                loss_s2 = criterion(output_s2, label_s)

            if self.num_c == 3:
                loss_s = loss_s1 + loss_s2 + loss_s3
            elif self.num_c == 4:
                loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4
            elif self.num_c == 5:
                loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_s5
            elif self.num_c == 6:
                loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_s5 + loss_s6
            elif self.num_c == 7:
                loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_s5 + loss_s6 + loss_s7
            else:
                loss_s = loss_s1 + loss_s2

            if self.num_c == 3:
                #add c3---------------------------------------------------
                loss_dis = self.discrepancy(output_t1, output_t2) + self.discrepancy(output_t2, output_t3)
                loss_dis = loss_dis + self.discrepancy(output_t1, output_t3)
                #----------------------------------------------------------
            elif self.num_c == 4:
                loss_dis = self.discrepancy(output_t1, output_t2) + self.discrepancy(output_t2, output_t3) + self.discrepancy(output_t3, output_t4)
                loss_dis = loss_dis + self.discrepancy(output_t1, output_t3) + self.discrepancy(output_t1, output_t4)
                loss_dis = loss_dis + self.discrepancy(output_t2, output_t4)
            #----------------------------------------------------------
            elif self.num_c == 5:
                loss_dis = self.discrepancy(output_t1, output_t2) + self.discrepancy(output_t2, output_t3) + self.discrepancy(output_t3, output_t4) + self.discrepancy(output_t4, output_t5)
                loss_dis = loss_dis + self.discrepancy(output_t1, output_t3) + self.discrepancy(output_t1, output_t4) + self.discrepancy(output_t1, output_t5)
                loss_dis = loss_dis + self.discrepancy(output_t2, output_t4) + self.discrepancy(output_t2, output_t5)
                loss_dis = loss_dis + self.discrepancy(output_t3, output_t5)
            #----------------------------------------------------------
            elif self.num_c == 6:
                loss_dis = self.discrepancy(output_t1, output_t2) + self.discrepancy(output_t2, output_t3) + self.discrepancy(output_t3, output_t4) + self.discrepancy(output_t4, output_t5) + self.discrepancy(output_t5, output_t6)
                loss_dis = loss_dis + self.discrepancy(output_t1, output_t3) + self.discrepancy(output_t1, output_t4) + self.discrepancy(output_t1, output_t5) + self.discrepancy(output_t1, output_t6)
                loss_dis = loss_dis + self.discrepancy(output_t2, output_t4) + self.discrepancy(output_t2, output_t5) + self.discrepancy(output_t2, output_t6)
                loss_dis = loss_dis + self.discrepancy(output_t3, output_t5) + self.discrepancy(output_t3, output_t6)
                loss_dis = loss_dis + self.discrepancy(output_t4, output_t6)
            # ----------------------------------------------------------
            elif self.num_c == 7:
                loss_dis = self.discrepancy(output_t1, output_t2) + self.discrepancy(output_t2, output_t3) + self.discrepancy(output_t3, output_t4) + self.discrepancy(output_t4, output_t5) + self.discrepancy(output_t5, output_t6) + self.discrepancy(output_t6, output_t7)
                loss_dis = loss_dis + self.discrepancy(output_t1, output_t3) + self.discrepancy(output_t1, output_t4) + self.discrepancy(output_t1, output_t5) + self.discrepancy(output_t1, output_t6) + self.discrepancy(output_t1, output_t7)
                loss_dis = loss_dis + self.discrepancy(output_t2, output_t4) + self.discrepancy(output_t2, output_t5) + self.discrepancy(output_t2, output_t6) + self.discrepancy(output_t2, output_t7)
                loss_dis = loss_dis + self.discrepancy(output_t3, output_t5) + self.discrepancy(output_t3, output_t6) + self.discrepancy(output_t3, output_t7)
                loss_dis = loss_dis + self.discrepancy(output_t4, output_t6) + self.discrepancy(output_t4, output_t7)
                loss_dis = loss_dis + self.discrepancy(output_t5, output_t7)
            # ----------------------------------------------------------
            else:
                loss_dis = self.discrepancy(output_t1, output_t2)

            loss = loss_s - loss_dis
            loss.backward()
            if self.num_c == 3:
                self.opt_c1.step()
                self.opt_c2.step()
                self.opt_c3.step()
            elif self.num_c == 4:
                self.opt_c1.step()
                self.opt_c2.step()
                self.opt_c3.step()
                self.opt_c4.step()
            elif self.num_c == 5:
                self.opt_c1.step()
                self.opt_c2.step()
                self.opt_c3.step()
                self.opt_c4.step()
                self.opt_c5.step()
            elif self.num_c == 6:
                self.opt_c1.step()
                self.opt_c2.step()
                self.opt_c3.step()
                self.opt_c4.step()
                self.opt_c5.step()
                self.opt_c6.step()
            elif self.num_c == 7:
                self.opt_c1.step()
                self.opt_c2.step()
                self.opt_c3.step()
                self.opt_c4.step()
                self.opt_c5.step()
                self.opt_c6.step()
                self.opt_c7.step()
            else:
                self.opt_c1.step()
                self.opt_c2.step()

            self.reset_grad()

            for i in xrange(self.num_k):
                #
                feat_t = self.G(img_t)
                if self.num_c == 3:
                    #add c3---------------------------------------------------
                    output_t1 = self.C1(feat_t)
                    output_t2 = self.C2(feat_t)
                    output_t3 = self.C3(feat_t)
                    #----------------------------------------------------------
                elif self.num_c == 4:
                    output_t1 = self.C1(feat_t)
                    output_t2 = self.C2(feat_t)
                    output_t3 = self.C3(feat_t)
                    output_t4 = self.C4(feat_t)
                    #----------------------------------------------------------
                elif self.num_c == 5:
                    output_t1 = self.C1(feat_t)
                    output_t2 = self.C2(feat_t)
                    output_t3 = self.C3(feat_t)
                    output_t4 = self.C4(feat_t)
                    output_t5 = self.C5(feat_t)
                    # ----------------------------------------------------------
                elif self.num_c == 6:
                    output_t1 = self.C1(feat_t)
                    output_t2 = self.C2(feat_t)
                    output_t3 = self.C3(feat_t)
                    output_t4 = self.C4(feat_t)
                    output_t5 = self.C5(feat_t)
                    output_t6 = self.C6(feat_t)
                    #  ----------------------------------------------------------
                elif self.num_c == 7:
                    output_t1 = self.C1(feat_t)
                    output_t2 = self.C2(feat_t)
                    output_t3 = self.C3(feat_t)
                    output_t4 = self.C4(feat_t)
                    output_t5 = self.C5(feat_t)
                    output_t6 = self.C6(feat_t)
                    output_t7 = self.C7(feat_t)
                    # ----------------------------------------------------------
                else:
                    output_t1 = self.C1(feat_t)
                    output_t2 = self.C2(feat_t)



                if self.num_c == 3:
                    #add c3---------------------------------------------------
                    loss_dis = self.discrepancy(output_t1, output_t2) + self.discrepancy(output_t2, output_t3)
                    loss_dis = loss_dis + self.discrepancy(output_t1, output_t3)
                    #----------------------------------------------------------
                elif self.num_c == 4:
                    loss_dis = self.discrepancy(output_t1, output_t2) + self.discrepancy(output_t2, output_t3) + self.discrepancy(output_t3, output_t4)
                    loss_dis = loss_dis + self.discrepancy(output_t1, output_t3) + self.discrepancy(output_t1, output_t4)
                    loss_dis = loss_dis + self.discrepancy(output_t2, output_t4)
                    #----------------------------------------------------------
                elif self.num_c == 5:
                    loss_dis = self.discrepancy(output_t1, output_t2) + self.discrepancy(output_t2, output_t3) + self.discrepancy(output_t3, output_t4) + self.discrepancy(output_t4, output_t5)
                    loss_dis = loss_dis + self.discrepancy(output_t1, output_t3) + self.discrepancy(output_t1, output_t4) + self.discrepancy(output_t1, output_t5)
                    loss_dis = loss_dis + self.discrepancy(output_t2, output_t4) + self.discrepancy(output_t2, output_t5)
                    loss_dis = loss_dis + self.discrepancy(output_t3, output_t5)
                    #----------------------------------------------------------
                elif self.num_c == 6:
                    loss_dis = self.discrepancy(output_t1, output_t2) + self.discrepancy(output_t2, output_t3) + self.discrepancy(output_t3, output_t4) + self.discrepancy(output_t4, output_t5) + self.discrepancy(output_t5, output_t6)
                    loss_dis = loss_dis + self.discrepancy(output_t1, output_t3) + self.discrepancy(output_t1, output_t4) + self.discrepancy(output_t1, output_t5) + self.discrepancy(output_t1, output_t6)
                    loss_dis = loss_dis + self.discrepancy(output_t2, output_t4) + self.discrepancy(output_t2, output_t5) + self.discrepancy(output_t2, output_t6)
                    loss_dis = loss_dis + self.discrepancy(output_t3, output_t5) + self.discrepancy(output_t3, output_t6)
                    loss_dis = loss_dis + self.discrepancy(output_t4, output_t6)
                    # ----------------------------------------------------------
                elif self.num_c == 7:
                    loss_dis = self.discrepancy(output_t1, output_t2) + self.discrepancy(output_t2, output_t3) + self.discrepancy(output_t3, output_t4) + self.discrepancy(output_t4, output_t5) + self.discrepancy(output_t5, output_t6) + self.discrepancy(output_t6, output_t7)
                    loss_dis = loss_dis + self.discrepancy(output_t1, output_t3) + self.discrepancy(output_t1, output_t4) + self.discrepancy(output_t1, output_t5) + self.discrepancy(output_t1, output_t6) + self.discrepancy(output_t1, output_t7)
                    loss_dis = loss_dis + self.discrepancy(output_t2, output_t4) + self.discrepancy(output_t2, output_t5) + self.discrepancy(output_t2, output_t6) + self.discrepancy(output_t2, output_t7)
                    loss_dis = loss_dis + self.discrepancy(output_t3, output_t5) + self.discrepancy(output_t3, output_t6) + self.discrepancy(output_t3, output_t7)
                    loss_dis = loss_dis + self.discrepancy(output_t4, output_t6) + self.discrepancy(output_t4, output_t7)
                    loss_dis = loss_dis + self.discrepancy(output_t5, output_t7)
                    # ----------------------------------------------------------
                else:
                    loss_dis = self.discrepancy(output_t1, output_t2)


                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()

            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                if self.num_c == 3:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss3: {:.6f}\t  Discrepancy: {:.6f}'.format(
                        epoch, batch_idx, 100,
                        100. * batch_idx / 70000, loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_s3.cpu().data.numpy(), loss_dis.cpu().data.numpy()))
                elif self.num_c == 4:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss3: {:.6f}\t Loss4: {:.6f}\t  Discrepancy: {:.6f}'.format(
                        epoch, batch_idx, 100,
                        100. * batch_idx / 70000, loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_s3.cpu().data.numpy(), loss_s4.cpu().data.numpy(), loss_dis.cpu().data.numpy()))
                    #----------------------------------------------------------
                elif self.num_c == 5:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss3: {:.6f}\t Loss4: {:.6f}\t Loss5: {:.6f}\t  Discrepancy: {:.6f}'.format(
                        epoch, batch_idx, 100,
                        100. * batch_idx / 70000, loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_s3.cpu().data.numpy(), loss_s4.cpu().data.numpy(), loss_s5.cpu().data.numpy(), loss_dis.cpu().data.numpy()))
                        #----------------------------------------------------------
                elif self.num_c == 6:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss3: {:.6f}\t Loss4: {:.6f}\t Loss5: {:.6f}\t Loss6: {:.6f}\t  Discrepancy: {:.6f}'.format(
                        epoch, batch_idx, 100,
                        100. * batch_idx / 70000, loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_s3.cpu().data.numpy(), loss_s4.cpu().data.numpy(), loss_s5.cpu().data.numpy(), loss_s6.cpu().data.numpy(), loss_dis.cpu().data.numpy()))
                            # ----------------------------------------------------------
                elif self.num_c == 7:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss3: {:.6f}\t Loss4: {:.6f}\t Loss5: {:.6f}\t Loss6: {:.6f}\t Loss7: {:.6f}\t  Discrepancy: {:.6f}'.format(
                        epoch, batch_idx, 100,
                        100. * batch_idx / 70000, loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_s3.cpu().data.numpy(), loss_s4.cpu().data.numpy(), loss_s5.cpu().data.numpy(), loss_s6.cpu().data.numpy(), loss_s7.cpu().data.numpy(), loss_dis.cpu().data.numpy()))
                                # ----------------------------------------------------------
                else:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                        epoch, batch_idx, 100,
                        100. * batch_idx / 70000, loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_dis.cpu().data.numpy()))
                if record_file:
                    record = open(record_file, 'a')
                    if self.num_c == 3:
                        record.write('%s %s %s %s\n' % (loss_dis.cpu().data.numpy(), loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_s3.cpu().data.numpy()))
                    elif self.num_c == 4:
                        record.write('%s %s %s %s %s\n' % (loss_dis.cpu().data.numpy(), loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_s3.cpu().data.numpy(), loss_s4.cpu().data.numpy()))
                    elif self.num_c == 5:
                        record.write('%s %s %s %s %s %s\n' % (loss_dis.cpu().data.numpy(), loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_s3.cpu().data.numpy(), loss_s4.cpu().data.numpy(), loss_s5.cpu().data.numpy()))
                    elif self.num_c == 6:
                        record.write('%s %s %s %s %s %s %s\n' % (loss_dis.cpu().data.numpy(), loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_s3.cpu().data.numpy(), loss_s4.cpu().data.numpy(), loss_s5.cpu().data.numpy(), loss_s6.cpu().data.numpy()))
                    elif self.num_c == 7:
                        record.write('%s %s %s %s %s %s %s %s\n' % (loss_dis.cpu().data.numpy(), loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_s3.cpu().data.numpy(), loss_s4.cpu().data.numpy(), loss_s5.cpu().data.numpy(), loss_s6.cpu().data.numpy(), loss_s7.cpu().data.numpy()))
                    else:
                        record.write('%s %s %s\n' % (loss_dis.cpu().data.numpy(), loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy()))
                    record.close()
        return batch_idx

    def train_onestep(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            label_s = Variable(label_s.long().cuda())
            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward(retain_variables=True)
            feat_t = self.G(img_t)
            self.C1.set_lambda(1.0)
            self.C2.set_lambda(1.0)
            output_t1 = self.C1(feat_t, reverse=True)
            output_t2 = self.C2(feat_t, reverse=True)
            loss_dis = -self.discrepancy(output_t1, output_t2)
            #loss_dis.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.opt_g.step()
            self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy(), loss_dis.cpu().data.numpy()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.cpu().data.numpy(), loss_s1.cpu().data.numpy(), loss_s2.cpu().data.numpy()))
                    record.close()
        return batch_idx

    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        self.C3.eval()
        self.C4.eval()
        self.C5.eval()
        self.C6.eval()
        self.C7.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        correct5 = 0
        correct6 = 0
        correct7 = 0
        correct8 = 0
        size = 0
        for batch_idx, data in enumerate(self.dataset_test):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            feat = self.G(img)
            output1 = self.C1(feat)
            output2 = self.C2(feat)
            if self.num_c > 2:
                #add c3---------------------------------------------------
                output3 = self.C3(feat)
                #----------------------------------------------------------
                if self.num_c > 3:
                    output4 = self.C4(feat)
                    #----------------------------------------------------------
                    if self.num_c > 4:
                        output5 = self.C5(feat)
                        #----------------------------------------------------------
                        if self.num_c > 5:
                            output6 = self.C6(feat)
                            # ----------------------------------------------------------
                            if self.num_c > 6:
                                output7 = self.C7(feat)
                                # ----------------------------------------------------------

            test_loss += F.nll_loss(output1, label).cpu().data.numpy()
            output_ensemble = output1 + output2
            pred2 = output2.data.max(1)[1]
            pred1 = output1.data.max(1)[1]

            if self.num_c > 2:
                #add c3---------------------------------------------------
                pred3 = output3.data.max(1)[1]
                #----------------------------------------------------------
                if self.num_c > 3:
                    pred4 = output4.data.max(1)[1]
                    #----------------------------------------------------------
                    if self.num_c > 4:
                        pred5 = output5.data.max(1)[1]
                        #----------------------------------------------------------
                        if self.num_c > 5:
                            pred6 = output6.data.max(1)[1]
                            # ----------------------------------------------------------
                            if self.num_c > 6:
                                pred7 = output7.data.max(1)[1]
                                # ----------------------------------------------------------

            pred_ensemble = output_ensemble.data.max(1)[1]
            k = label.data.size()[0]

            correct1 += pred1.eq(label.data).cpu().sum()
            correct2 += pred2.eq(label.data).cpu().sum()
            if self.num_c > 2:
                #add c3---------------------------------------------------
                correct3 += pred3.eq(label.data).cpu().sum()
                #----------------------------------------------------------
                if self.num_c > 3:
                    correct4 += pred4.eq(label.data).cpu().sum()
                    #----------------------------------------------------------
                    if self.num_c > 4:
                        correct5 += pred5.eq(label.data).cpu().sum()
                        #----------------------------------------------------------
                        if self.num_c > 5:
                            correct6 += pred6.eq(label.data).cpu().sum()
                            # ----------------------------------------------------------
                            if self.num_c > 6:
                                correct7 += pred7.eq(label.data).cpu().sum()
                                # ----------------------------------------------------------

            correct8 += pred_ensemble.eq(label.data).cpu().sum()
            size += k

        test_loss = test_loss / size

        if self.num_c == 3:
            print(
                '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy C3: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                    test_loss, correct1, size,
                    100. * correct1 / size, correct2, size, 100. * correct2 / size,
                    correct3, size, 100. * correct3 / size,
                    correct8, size, 100. * correct8 / size))
        elif self.num_c == 4:
            print(
                '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy C3: {}/{} ({:.0f}%) Accuracy C4: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                    test_loss, correct1, size,
                    100. * correct1 / size, correct2, size, 100. * correct2 / size,
                    correct3, size, 100. * correct3 / size,
                    correct4, size, 100. * correct4 / size,
                    correct8, size, 100. * correct8 / size))
            #----------------------------------------------------------
        elif self.num_c == 5:
            print(
                '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy C3: {}/{} ({:.0f}%) Accuracy C4: {}/{} ({:.0f}%) Accuracy C5: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                    test_loss, correct1, size,
                    100. * correct1 / size, correct2, size, 100. * correct2 / size,
                    correct3, size, 100. * correct3 / size,
                    correct4, size, 100. * correct4 / size,
                    correct5, size, 100. * correct5 / size,
                    correct8, size, 100. * correct8 / size))
            #----------------------------------------------------------
        elif self.num_c == 6:
            print(
                '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy C3: {}/{} ({:.0f}%) Accuracy C4: {}/{} ({:.0f}%) Accuracy C5: {}/{} ({:.0f}%) Accuracy C6: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                    test_loss, correct1, size,
                    100. * correct1 / size, correct2, size, 100. * correct2 / size,
                    correct3, size, 100. * correct3 / size,
                    correct4, size, 100. * correct4 / size,
                    correct5, size, 100. * correct5 / size,
                    correct6, size, 100. * correct6 / size,
                    correct8, size, 100. * correct8 / size))
            # ----------------------------------------------------------
        elif self.num_c == 7:
            print(
                '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy C3: {}/{} ({:.0f}%) Accuracy C4: {}/{} ({:.0f}%) Accuracy C5: {}/{} ({:.0f}%) Accuracy C6: {}/{} ({:.0f}%) Accuracy C7: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                    test_loss, correct1, size,
                    100. * correct1 / size, correct2, size, 100. * correct2 / size,
                    correct3, size, 100. * correct3 / size,
                    correct4, size, 100. * correct4 / size,
                    correct5, size, 100. * correct5 / size,
                    correct6, size, 100. * correct6 / size,
                    correct7, size, 100. * correct7 / size,
                    correct8, size, 100. * correct8 / size))
            # ----------------------------------------------------------
        else:
            print(
                '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                    test_loss, correct1, size,
                    100. * correct1 / size, correct2, size, 100. * correct2 / size, correct8, size, 100. * correct8 / size))

        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.G,
                       '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C1,
                       '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C2,
                       '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write('number of classifiers: %s\n' % self.num_c)
            record.write('%s %s %s %s %s %s %s %s\n' % (float(correct1) / size, float(correct2) / size, float(correct3) / size, float(correct4) / size, float(correct5) / size, float(correct6) / size, float(correct7) / size, float(correct8) / size))
            record.close()
